from pathlib import Path

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

__all__ = [
    "load_dataset",
]

_labels = [
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
transform = T.ToTensor()
target_transform = lambda x: _labels[x]


def load_dataset(
    root: Path,
    train: bool,
) -> CIFAR10:
    return CIFAR10(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
