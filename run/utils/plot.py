import torch
from matplotlib.axes import Axes
from torch import Tensor

__all__ = [
    "quantiles",
]


def quantiles(
    ax: Axes,
    data: Tensor,
    color: str,
    dim: int = 0,
    alpha: float = 0.1,
    label: str = None,
) -> None:
    y = torch.quantile(
        data,
        torch.tensor(
            [
                # 0,
                0.07,
                0.16,
                0.31,
                0.5,
                0.69,
                0.84,
                0.93,
                # 1.0,
            ]
        ),
        dim=dim,
    )
    x = range(len(y[0]))

    # Plot median line
    ax.plot(x, y[3], "-", color=color, label=label)

    # Plot shaded regions for different percentiles
    for i in range(4):
        ax.fill_between(x, y[i], y[-i - 1], alpha=alpha, color=color)
