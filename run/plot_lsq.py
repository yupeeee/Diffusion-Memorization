import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import DATA_DIR, FIG_DIR
from torch import Tensor


def compute_coefs(num_inference_steps: int) -> Tuple[List[float], List[float]]:
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas = pipe.alphas_cumprod(num_inference_steps)

    # Pre-compute sqrt terms
    sqrt_alpha = alphas.sqrt()
    sqrt_1_minus_alpha = (1 - alphas).sqrt()
    sqrt_alpha_prev = alphas[1:].sqrt()
    sqrt_1_minus_alpha_prev = (1 - alphas[1:]).sqrt()

    # Calculate coefficients
    image_coefs = (
        sqrt_alpha_prev * sqrt_1_minus_alpha[:-1]
        - sqrt_alpha[:-1] * sqrt_1_minus_alpha_prev
    ) / sqrt_1_minus_alpha[:-1]
    noise_coefs = (
        sqrt_1_minus_alpha[:-1] - sqrt_1_minus_alpha_prev
    ) / sqrt_1_minus_alpha[:-1]
    coefs = list(zip(image_coefs.tolist(), noise_coefs.tolist()))

    # Adjust coefficients for x_0 and x_T
    coefs[0] = (coefs[0][0], 1 - coefs[0][1])
    for i in range(len(coefs) - 1):
        coefs[i + 1] = (
            coefs[i + 1][0] + (1 - coefs[i + 1][1]) * coefs[i][0],
            (1 - coefs[i + 1][1]) * coefs[i][1],
        )

    image_coefs = [c[0] for c in coefs]
    noise_coefs = [c[1] for c in coefs]

    return image_coefs, noise_coefs


def plot_lsq(
    lsq_0: Tensor,
    lsq_t: Tensor,
    i: int,
) -> None:
    w_0s = [lsq_0["x_0@x_t"][i], lsq_t["x_0@x_t"][i]]
    w_Ts = [lsq_0["x_T@x_t"][i], lsq_t["x_T@x_t"][i]]
    errs = [lsq_0["err"][i], lsq_t["err"][i]]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    for j, (ax, w_0, w_T, err) in enumerate(zip(axes, w_0s, w_Ts, errs)):
        ax.fill_between(
            range(len(w_0)),
            w_0,
            alpha=0.5,
            color="blue",
            label="$\\mathbf{{x}}_{{0}}$@$\\mathbf{{x}}_{{t}}$" if j == 0 else "$\\hat{{\\mathbf{{x}}}}_{{0}}$@$\\mathbf{{x}}_{{t}}$",
        )
        ax.fill_between(
            range(len(w_T)),
            w_0,
            [sum(x) for x in zip(w_0, w_T)],
            alpha=0.5,
            color="red",
            label="$\\mathbf{{x}}_{{T}}$@$\\mathbf{{x}}_{{t}}$",
        )
        ax.plot(
            range(1, len(image_coefs) + 1),
            image_coefs,
            "--",
            color="blue",
            label="$\\omega_{{0}}^{{(t)}}$",
        )
        ax.plot(
            range(1, len(noise_coefs) + 1),
            [sum(x) for x in zip(image_coefs, noise_coefs)],
            "--",
            color="red",
            label="$\\omega_{{T}}^{{(t)}}$",
        )
        ax.set_xlabel("$t$")
        ax.set_xticks([0, args.num_inference_steps], ["$T$", "$0$"])
        ax.legend(loc="lower right", fontsize=10)
    axes[0].set_ylabel("Portion@$x_{{t}}$")
    y_min = min(
        min(w_0s[0].min(), w_0s[1].min()),
        min((w_0s[0] + w_Ts[0]).min(), (w_0s[1] + w_Ts[1]).min())
    )
    y_max = max(
        max((w_0s[0] + w_Ts[0]).max(), (w_0s[1] + w_Ts[1]).max()),
        max(w_0s[0].max(), w_0s[1].max())
    )
    axes[0].set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        labels=["0", "0.25", "0.50", "0.75", "1.00", "1.25", "1.50"],
    )
    axes[1].set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        labels=[""] * 7,
    )
    for ax in axes:
        ax.set_ylim(min(-0.05, y_min), max(1.5, y_max))
        ax.grid(alpha=0.25)
    plt.savefig(
        os.path.join(save_dir, f"{i}.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-inference-steps", type=int, required=False, default=50)
    parser.add_argument("--seed", type=int, required=False, default=0)
    args = parser.parse_args()

    # Compute image/noise coeffs
    image_coefs, noise_coefs = compute_coefs(args.num_inference_steps)

    # Load data
    lsq_0 = torch.load(
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"seed_{args.seed}",
            f"lsq_0.pt",
        ),
        weights_only=True,
    )
    lsq_t = torch.load(
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"seed_{args.seed}",
            f"lsq_t.pt",
        ),
        weights_only=True,
    )

    # Plot
    plt.rcParams["font.size"] = 15
    save_dir = os.path.join(FIG_DIR, "sdv1-memorization", f"seed_{args.seed}", "lsq")
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm.trange(1000, desc="Plotting LSQ"):
        plot_lsq(lsq_0, lsq_t, i)
