"""MLflow tracking-server setup with a local SQLite fallback if the remote server is unreachable."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FALLBACK_URI = "sqlite:///" + str((PROJECT_ROOT / "mlruns.db").as_posix())


def setup_mlflow(mlflow_config: Dict[str, Any]) -> bool:
    """Point mlflow at the remote tracking server, falling back to a local SQLite store if unreachable.

    MLflow 3.x puts the legacy `file:./mlruns` backend into maintenance mode (it now raises
    unless MLFLOW_ALLOW_FILE_STORE is set), so the local fallback uses a SQLite-backed store instead.

    Args:
        mlflow_config: The "mlflow" section of the env config (tracking_uri, experiment_name, ...).

    Returns:
        True if the remote tracking server is reachable and in use, False if we fell back to local.
    """
    import mlflow
    import requests

    tracking_uri = mlflow_config["tracking_uri"]
    experiment_name = mlflow_config["experiment_name"]

    remote_ok = False
    try:
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        remote_ok = response.status_code == 200
    except requests.exceptions.RequestException as exc:
        warnings.warn(
            f"MLflow server at {tracking_uri} unreachable ({exc}). "
            f"Falling back to local store: {LOCAL_FALLBACK_URI}"
        )

    mlflow.set_tracking_uri(tracking_uri if remote_ok else LOCAL_FALLBACK_URI)
    mlflow.set_experiment(experiment_name)
    return remote_ok
