import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import DATA_DIR, FIG_DIR


def distribution_plot(data, save_path: Path):
    mem_data = data[:500].reshape(-1, args.num_inference_steps).transpose(0, 1)
    nor_data = data[500:].reshape(-1, args.num_inference_steps).transpose(0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    for d, color in [(mem_data, "red"), (nor_data, "blue")]:
        quantiles = torch.quantile(
            d,
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
            dim=1,
        )
        x = range(len(quantiles[0]))

        # Plot median line
        ax.plot(x, quantiles[3], "-", color=color)

        # Plot shaded regions for different percentiles
        for i in range(4):
            alpha = 0.1
            ax.fill_between(
                x, quantiles[i], quantiles[-i - 1], alpha=alpha, color=color
            )
    ax.set_xlabel("$t$")
    ax.set_xticks(
        range(0, args.num_inference_steps + 1, 10),
        labels=range(args.num_inference_steps, -1, -10),
    )
    ax.set_ylabel("Error")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=False, default=50)
    parser.add_argument("--num-seeds", type=int, required=False, default=50)
    args = parser.parse_args()

    for plot_type in args.plot.split(","):
        mses = []
        coss = []

        for seed in range(args.num_seeds):
            mses.append(
                torch.load(
                    os.path.join(
                        DATA_DIR,
                        "sdv1-memorization",
                        f"seed_{seed}",
                        f"mses_{plot_type}.pt",
                    ),
                    weights_only=True,
                )
            )
            coss.append(
                torch.load(
                    os.path.join(
                        DATA_DIR,
                        "sdv1-memorization",
                        f"seed_{seed}",
                        f"coss_{plot_type}.pt",
                    ),
                    weights_only=True,
                )
            )

        mses = torch.stack(mses, dim=0).transpose(0, 1)
        coss = torch.stack(coss, dim=0).transpose(0, 1)

        distribution_plot(
            mses, os.path.join(FIG_DIR, "sdv1-memorization", f"mses_{plot_type}.png")
        )
        distribution_plot(
            coss, os.path.join(FIG_DIR, "sdv1-memorization", f"coss_{plot_type}.png")
        )
