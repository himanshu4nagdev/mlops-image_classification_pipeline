"""Standalone evaluation of a trained model on the CIFAR-10 test set."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config, resolve_device
from src.data_loader import get_class_names, get_dataloaders
from src.mlflow_utils import setup_mlflow


def load_model_for_eval(model_source: str, from_mlflow: bool, device: torch.device) -> torch.nn.Module:
    """Load a trained model either from a local path or an MLflow model URI.

    Args:
        model_source: Local directory path, or an MLflow URI (e.g. "runs:/<run_id>/model",
            "models:/<name>/<stage_or_version>") when from_mlflow=True.
        from_mlflow: Whether model_source is an MLflow model URI.
        device: Device to move the model to.
    """
    if from_mlflow:
        import mlflow.pytorch

        model = mlflow.pytorch.load_model(model_source)
    else:
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(model_source)
    return model.to(device)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Run evaluation over the test set, returning accuracy, classification report, and confusion matrix."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(cm)

    metrics: Dict[str, Any] = {
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def log_eval_metrics_to_mlflow(metrics: Dict[str, Any], class_names: List[str], run_id: Optional[str]) -> None:
    """Log evaluation metrics into an existing MLflow run (by run_id), or a new run if none given."""
    import mlflow

    try:
        context = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name="evaluation")
        with context:
            mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
            for cls in class_names:
                cls_report = metrics["classification_report"].get(cls)
                if cls_report is None:
                    continue
                mlflow.log_metric(f"precision_{cls}", cls_report["precision"])
                mlflow.log_metric(f"recall_{cls}", cls_report["recall"])
                mlflow.log_metric(f"f1_{cls}", cls_report["f1-score"])
    except Exception as exc:
        warnings.warn(f"Failed to log evaluation metrics to MLflow: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 classifier.")
    parser.add_argument("--profile", type=str, default=None, help="Override active_profile from env_config.yaml")
    parser.add_argument("--model-path", type=str, default=None, help="Local path to a trained model directory")
    parser.add_argument("--model-uri", type=str, default=None, help="MLflow model URI, e.g. runs:/<id>/model")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run_id to log eval metrics into")
    args = parser.parse_args()

    if not args.model_path and not args.model_uri:
        parser.error("Provide either --model-path or --model-uri")

    cfg = load_config(profile_override=args.profile)
    device_ = resolve_device(cfg["device"])
    _, test_loader_ = get_dataloaders(cfg)
    class_names_ = get_class_names()

    if args.model_uri:
        model_ = load_model_for_eval(args.model_uri, from_mlflow=True, device=device_)
    else:
        model_ = load_model_for_eval(args.model_path, from_mlflow=False, device=device_)

    eval_metrics = evaluate_model(model_, test_loader_, class_names_, device_)

    setup_mlflow(cfg["mlflow"])
    log_eval_metrics_to_mlflow(eval_metrics, class_names_, args.run_id)
