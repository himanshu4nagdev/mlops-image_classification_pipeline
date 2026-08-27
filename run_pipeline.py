"""Orchestrator: load data -> train -> evaluate -> register model in MLflow.

Usage:
    python run_pipeline.py [--profile PROFILE] [--skip-train] [--skip-eval]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolve_device
from src.data_loader import get_class_names, get_dataloaders
from src.evaluate import evaluate_model, log_eval_metrics_to_mlflow
from src.mlflow_utils import setup_mlflow
from src.train import train


def print_profile_settings(config: Dict[str, Any]) -> None:
    print(f"\n=== Active profile: {config['profile_name']} ===")
    for key, value in config.items():
        if key in ("profile_name", "mlflow"):
            continue
        print(f"  {key}: {value}")
    print(f"  mlflow.tracking_uri: {config['mlflow']['tracking_uri']}")
    print(f"  mlflow.experiment_name: {config['mlflow']['experiment_name']}")
    print(f"  mlflow.registry_model_name: {config['mlflow']['registry_model_name']}")
    print("=" * 40)


def register_model(run_id: str, registry_model_name: str) -> None:
    """Register the trained model artifact from a run into the MLflow model registry."""
    import mlflow

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=registry_model_name)
    print(f"Registered model '{registry_model_name}' version {result.version} from {model_uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the image classification training pipeline.")
    parser.add_argument("--profile", type=str, default=None, help="Override active_profile from env_config.yaml")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation stage")
    args = parser.parse_args()

    config = load_config(profile_override=args.profile)
    print_profile_settings(config)

    device = resolve_device(config["device"])
    print(f"Resolved device: {device}")

    class_names = get_class_names()
    run_id = None
    model = None

    if not args.skip_train:
        model, train_metrics = train(config)
        run_id = train_metrics.get("run_id")
        print(f"Training complete. MLflow run_id: {run_id}")
    else:
        print("Skipping training (--skip-train). Evaluation/registration require a model to already exist.")

    if not args.skip_eval:
        if model is None:
            raise RuntimeError("Cannot evaluate: no model available. Re-run without --skip-train, "
                                "or extend this script to load a model by path/URI.")
        _, test_loader = get_dataloaders(config)
        eval_metrics = evaluate_model(model, test_loader, class_names, device)
        setup_mlflow(config["mlflow"])
        log_eval_metrics_to_mlflow(eval_metrics, class_names, run_id)
    else:
        print("Skipping evaluation (--skip-eval).")

    if run_id:
        register_model(run_id, config["mlflow"]["registry_model_name"])

        import mlflow

        experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
        experiment_id = experiment.experiment_id if experiment else "0"
        run_url = f"{config['mlflow']['tracking_uri']}/#/experiments/{experiment_id}/runs/{run_id}"
        print("\n=== Pipeline summary ===")
        print(f"Profile: {config['profile_name']}")
        print(f"MLflow run_id: {run_id}")
        print(f"MLflow run URL: {run_url}")
        print(f"Registered model: {config['mlflow']['registry_model_name']}")
    else:
        print("\nNo run_id produced (training was skipped) — nothing registered.")


if __name__ == "__main__":
    main()
