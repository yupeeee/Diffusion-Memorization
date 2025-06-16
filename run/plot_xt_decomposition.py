import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import FIG_DIR, LOG_DIR
from torch import Tensor


def lstsq(
    args,
    x_ts: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    T = args.num_inference_steps
    x_ts = x_ts.reshape(T + 1, -1)  # (T + 1, dim)
    x_T = x_ts[0].unsqueeze(dim=0).repeat(T, 1)  # (T, dim)
    x_0 = x_ts[-1].unsqueeze(dim=0).repeat(T, 1)  # (T, dim)
    X = x_ts[1:]  # (T, dim)

    # X ~ a * x_0 + b * x_T
    lstsq = torch.linalg.lstsq(
        torch.stack([x_0, x_T], dim=2),
        X[..., None],
    ).solution

    A, B = lstsq[:, 0, 0], lstsq[:, 1, 0]  # (T,)

    X_hat = A[:, None] * x_0[None, :] + B[:, None] * x_T[None, :]

    residuals = (X - X_hat).reshape(T, -1)
    errs = torch.sqrt(torch.mean(residuals**2, dim=1))
    return (A, B, errs)


if __name__ == "__main__":
    args = utils.args.load()

    prompt_idxs = list(range(0, 16)) + list(range(500, 516))

    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas_cumprod = pipe.alphas_cumprod(args.num_inference_steps)
    w_0s = alphas_cumprod.sqrt()
    w_Ts = (1 - alphas_cumprod).sqrt()

    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Plotting x_t decomposition...",
    ):
        # Load latents
        x_ts = torch.load(
            os.path.join(
                LOG_DIR,
                "sdv1-memorization",
                f"step{args.num_inference_steps}-guid{args.guidance_scale}",
                f"seed_{args.seed}",
                "latents",
                f"{prompt_idx}.pt",
            ),
            weights_only=True,
        )

        # Decompose latents
        A, B, errs = lstsq(args, x_ts)

        # x_0, x_T/4, x_T/2, x_3T/4, x_T to image
        images = pipe.decode(
            torch.stack(
                [
                    x_ts[0],
                    x_ts[len(x_ts) // 4],
                    x_ts[len(x_ts) // 2],
                    x_ts[3 * len(x_ts) // 4],
                    x_ts[-1],
                ],
                dim=0,
            )
        )

        # Plot
        plt.rcParams["font.size"] = 15
        fig, ax1 = plt.subplots(figsize=(3, 3))

        # Left y-axis for portions
        ax1.plot(
            range(1, len(A) + 1),
            A,
            "-",
            color="blue",
            label="$\\mathbf{{x}}_{{0}}$@$\\mathbf{{x}}_{{t}}$",
            zorder=2,
        )
        ax1.plot(
            range(1, len(B) + 1),
            B,
            "-",
            color="red",
            label="$\\mathbf{{x}}_{{T}}$@$\\mathbf{{x}}_{{t}}$",
            zorder=2,
        )
        ax1.plot(
            range(1, len(w_0s) + 1),
            w_0s,
            "--",
            color="blue",
            label="$\\sqrt{{\\alpha_{{t}}}}$",
            zorder=2,
        )
        ax1.plot(
            range(1, len(w_Ts) + 1),
            w_Ts,
            "--",
            color="red",
            label="$\\sqrt{{1 - \\alpha_{{t}}}}$",
            zorder=2,
        )
        ax1.set_xlabel("$t$")
        ax1.set_xlim(-2.5, args.num_inference_steps + 2.5)
        ax1.set_xticks([0, args.num_inference_steps], ["$T$", "$0$"])
        ax1.set_ylabel("Portion@$x_{{t}}$")
        ax1.set_ylim(-0.05, 1.25)
        ax1.set_yticks(
            [0, 0.25, 0.5, 0.75, 1.0],
            labels=["0", "0.25", "0.50", "0.75", "1"],
        )
        ax1.grid(alpha=0.25)
        # Right y-axis for errors
        ax2 = ax1.twinx()
        ax2.bar(
            range(1, len(errs) + 1),
            errs,
            color="black",
            alpha=0.1,
            width=1.0,
            label="RMSE",
            zorder=0,
        )
        ax2.set_ylabel("RMSE", color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.set_ylim(-0.05, 1.25)
        ax2.set_yticks(
            [0, 0.25, 0.5, 0.75, 1.0],
            labels=["0", "0.25", "0.50", "0.75", "1"],
        )

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

        fig.figimage(images[0].resize((80, 80)), 295, 765, zorder=1)
        fig.figimage(images[1].resize((80, 80)), 433, 765, zorder=1)
        fig.figimage(images[2].resize((80, 80)), 571, 765, zorder=1)
        fig.figimage(images[3].resize((80, 80)), 709, 765, zorder=1)
        fig.figimage(images[4].resize((80, 80)), 848, 765, zorder=1)
        save_dir = os.path.join(
            FIG_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "xt_decomposition",
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{prompt_idx}-{args.seed}.pdf"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
