"""Training loop for MobileNetV2 on CIFAR-10 with MLflow tracking."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForImageClassification

from src.config import load_config, print_gpu_memory, resolve_device
from src.data_loader import get_class_names, get_dataloaders
from src.mlflow_utils import setup_mlflow

MODEL_CHECKPOINT = "google/mobilenet_v2_1.0_224"


def build_model(num_labels: int) -> nn.Module:
    """Load MobileNetV2 from transformers, replacing the classification head for CIFAR-10."""
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    device: torch.device,
    fp16: bool,
    grad_accum_steps: int,
) -> Tuple[float, float]:
    """Run one training epoch with mixed precision and gradient accumulation. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=fp16):
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss / grad_accum_steps

        if fp16 and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if fp16 and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item() * images.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train MobileNetV2 on CIFAR-10 per the given config profile, logging to MLflow.

    Args:
        config: Flattened profile dict from config.load_config() (includes "mlflow" section).

    Returns:
        (trained_model, metrics) where metrics includes final train loss/accuracy and the
        MLflow run_id (needed downstream for evaluation/registration).
    """
    import mlflow
    import mlflow.pytorch

    device = resolve_device(config["device"])
    print(f"Using device: {device}")

    class_names = get_class_names()
    train_loader, _ = get_dataloaders(config)

    print_gpu_memory("before model load")
    model = build_model(num_labels=len(class_names)).to(device)
    print_gpu_memory("after model load")

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["fp16"] and device.type == "cuda")

    remote_ok = setup_mlflow(config["mlflow"])
    print(f"MLflow tracking: {'remote' if remote_ok else 'local fallback (./mlruns)'}")

    run_name = f"{config['profile_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    metrics: Dict[str, Any] = {}
    run_id = ""

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        try:
            loggable_params = {k: v for k, v in config.items() if k != "mlflow"}
            mlflow.log_params(loggable_params)
            mlflow.log_param("active_profile", config["profile_name"])
        except Exception as exc:
            warnings.warn(f"Failed to log params to MLflow: {exc}")

        for epoch in range(1, config["max_epochs"] + 1):
            epoch_start = time.time()
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                fp16=config["fp16"],
                grad_accum_steps=config["gradient_accumulation_steps"],
            )
            scheduler.step()
            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{config['max_epochs']} "
                f"- loss: {train_loss:.4f} - acc: {train_acc:.4f} - {epoch_time:.1f}s"
            )
            try:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
            except Exception as exc:
                warnings.warn(f"Failed to log metrics to MLflow: {exc}")

            metrics["train_loss"] = train_loss
            metrics["train_accuracy"] = train_acc

        print_gpu_memory("after training")

        try:
            # artifact_path= (not name=): this mlflow install is pinned to 2.18.0 to match the
            # tracking server, and log_model()'s signature in the 2.x line uses artifact_path
            # as the required positional/keyword arg. "name=" is a 3.x-only parameter and
            # raises "missing 1 required positional argument: 'artifact_path'" on 2.x.
            #
            # No serialization_format=: that kwarg was added in mlflow 3.x's log_model()/save().
            # On 2.18.0, save() doesn't accept it at all and raises "unexpected keyword
            # argument 'serialization_format'". mlflow 2.x's default pytorch save already uses
            # plain pickling, which is what we want for this HF model anyway (its forward()
            # takes keyword args like pixel_values/labels, so the pt2/dynamo-trace format some
            # newer mlflow defaults toward would not apply here even if it were available).
            mlflow.pytorch.log_model(model, artifact_path="model")
        except Exception as exc:
            warnings.warn(f"Failed to log model artifact to MLflow: {exc}")

    metrics["run_id"] = run_id
    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MobileNetV2 on CIFAR-10.")
    parser.add_argument("--profile", type=str, default=None, help="Override active_profile from env_config.yaml")
    args = parser.parse_args()

    cfg = load_config(profile_override=args.profile)
    print(f"Training with profile: {cfg['profile_name']}")
    trained_model, final_metrics = train(cfg)
    print(f"Final metrics: {final_metrics}")