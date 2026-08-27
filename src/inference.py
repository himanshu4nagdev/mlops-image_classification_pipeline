"""Single-image inference against a model pulled from the MLflow model registry.

Local testing only — not a production serving path (that lives on the remote server).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.config import load_config, resolve_device
from src.data_loader import build_transforms, get_class_names
from src.mlflow_utils import setup_mlflow


def resolve_model_uri(registry_model_name: str) -> str:
    """Resolve the model URI for the latest 'Production' version, falling back to the latest 'None'-stage version."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    for stage in ("Production", "None"):
        try:
            versions = client.get_latest_versions(registry_model_name, stages=[stage])
        except Exception:
            versions = []
        if versions:
            print(f"Using registered model '{registry_model_name}' stage='{stage}' (version {versions[0].version})")
            return f"models:/{registry_model_name}/{stage}"

    all_versions = client.search_model_versions(f"name='{registry_model_name}'")
    if not all_versions:
        raise ValueError(f"No versions found for registered model '{registry_model_name}'")
    latest = max(all_versions, key=lambda v: int(v.version))
    print(f"Using registered model '{registry_model_name}' latest version {latest.version} (no stage set)")
    return f"models:/{registry_model_name}/{latest.version}"


def load_and_preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    """Load an image file and apply the eval preprocessing pipeline, returning a batched tensor."""
    image = Image.open(image_path).convert("RGB")
    transform = build_transforms(image_size, train=False)
    tensor = transform(image)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_top_k(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: List[str],
    device: torch.device,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Run inference and return the top-k (class_name, confidence) predictions."""
    model.eval()
    image_tensor = image_tensor.to(device)
    outputs = model(pixel_values=image_tensor)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    top_probs, top_indices = probs.topk(k)
    return [(class_names[idx], prob.item()) for idx, prob in zip(top_indices.tolist(), top_probs)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-image inference using the MLflow model registry.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--profile", type=str, default=None, help="Override active_profile from env_config.yaml")
    args = parser.parse_args()

    cfg = load_config(profile_override=args.profile)
    device_ = resolve_device(cfg["device"])

    setup_mlflow(cfg["mlflow"])

    import mlflow.pytorch

    model_uri = resolve_model_uri(cfg["mlflow"]["registry_model_name"])
    model_ = mlflow.pytorch.load_model(model_uri).to(device_)

    image_tensor_ = load_and_preprocess_image(args.image_path, cfg["image_size"])
    predictions = predict_top_k(model_, image_tensor_, get_class_names(), device_, k=3)

    print(f"\nTop-3 predictions for {args.image_path}:")
    for rank, (label, confidence) in enumerate(predictions, start=1):
        print(f"  {rank}. {label:<12} {confidence * 100:.2f}%")
