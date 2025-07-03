import torch
from torch import Tensor

__all__ = [
    "mse",
    "cos",
]


def mse(
    x: Tensor,
    y: Tensor,
    dim: int = -1,
) -> Tensor:
    return ((x - y) ** 2).mean(dim=dim)


def cos(
    x: Tensor,
    y: Tensor,
    dim: int = -1,
) -> Tensor:
    return torch.nn.functional.cosine_similarity(x, y, dim=dim)
