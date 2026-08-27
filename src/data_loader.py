"""CIFAR-10 dataset loading and preprocessing into PyTorch DataLoaders."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_class_names() -> List[str]:
    """Return the CIFAR-10 class labels in index order."""
    return CIFAR10_CLASSES


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Build the torchvision transform pipeline for train or eval."""
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class CIFAR10TorchDataset(Dataset):
    """Wraps a Hugging Face CIFAR-10 split as a PyTorch Dataset, applying torchvision transforms."""

    def __init__(self, hf_split: Any, transform: transforms.Compose) -> None:
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        example = self.hf_split[idx]
        image = example["img"].convert("RGB")
        label = int(example["label"])
        return self.transform(image), label


def load_cifar10_splits() -> Any:
    """Load the CIFAR-10 dataset from Hugging Face, handling both current and legacy dataset names."""
    from datasets import load_dataset

    try:
        return load_dataset("uoft-cs/cifar10")
    except Exception:
        return load_dataset("cifar10")


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders using batch size, workers, etc. from the active config profile.

    Args:
        config: The flattened profile dict from config.load_config().

    Returns:
        (train_loader, test_loader)
    """
    raw = load_cifar10_splits()

    train_transform = build_transforms(config["image_size"], train=True)
    eval_transform = build_transforms(config["image_size"], train=False)

    train_dataset = CIFAR10TorchDataset(raw["train"], train_transform)
    test_dataset = CIFAR10TorchDataset(raw["test"], eval_transform)

    loader_kwargs: Dict[str, Any] = {
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
    }
    # prefetch_factor is only valid when num_workers > 0
    if config["num_workers"] > 0 and config.get("prefetch_factor") is not None:
        loader_kwargs["prefetch_factor"] = config["prefetch_factor"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config

    cfg = load_config()
    print(f"Loading CIFAR-10 with profile '{cfg['profile_name']}'...")
    train_dl, test_dl = get_dataloaders(cfg)
    print(f"Train batches: {len(train_dl)} (batch_size={cfg['batch_size']})")
    print(f"Test batches: {len(test_dl)} (batch_size={cfg['eval_batch_size']})")
    images, labels = next(iter(train_dl))
    print(f"Sample batch shape: {images.shape}, labels: {labels[:8].tolist()}")
    print(f"Class names: {get_class_names()}")
