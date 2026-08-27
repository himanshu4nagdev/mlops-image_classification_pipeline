"""Environment-profile config loading and device/GPU utilities shared across the pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "env_config.yaml"


def load_config(profile_override: Optional[str] = None, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load env_config.yaml and flatten the selected profile into a single dict.

    Args:
        profile_override: If set, use this profile instead of `active_profile` in the YAML.
        config_path: Optional override for the config file location.

    Returns:
        Dict containing all profile keys (device, batch_size, ...) plus
        "profile_name" and "mlflow" (the mlflow section, unchanged).
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    profile_name = profile_override or raw["active_profile"]
    profiles = raw.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")

    profile = dict(profiles[profile_name])
    profile["profile_name"] = profile_name
    profile["mlflow"] = raw["mlflow"]
    return profile


def resolve_device(requested_device: str):
    """Resolve the requested device string to a torch.device, falling back to CPU with a warning."""
    import torch

    if requested_device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested in config but not available on this machine — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def print_gpu_memory(stage: str) -> None:
    """Print free/used/total GPU memory (MiB) for device 0, if CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        print(f"[GPU MEM] {stage}: CUDA not available, skipping")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mib = free_bytes / (1024 ** 2)
    total_mib = total_bytes / (1024 ** 2)
    used_mib = total_mib - free_mib
    print(f"[GPU MEM] {stage}: used={used_mib:.0f} MiB, free={free_mib:.0f} MiB, total={total_mib:.0f} MiB")
