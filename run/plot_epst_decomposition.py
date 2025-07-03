import os

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import DATA_DIR, FIG_DIR

if __name__ == "__main__":
    args = utils.args.load()

    # Load noise prediction decompositions
    data = torch.load(
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "epst_decomposition.pt",
        ),
        weights_only=True,
    )
    _w_0s, _w_ts, w_0s, w_ts, errs = (
        data["_w_0"],
        data["_w_t"],
        data["w_0"],
        data["w_t"],
        data["err"],
    )

    prompt_idxs = list(range(0, 1000))
    save_dir = os.path.join(
        FIG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
        "epst_decomposition",
    )
    os.makedirs(save_dir, exist_ok=True)

    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Plotting noise prediction decompositions...",
    ):
        # Plot
        plt.rcParams["font.size"] = 15
        fig, ax = plt.subplots(figsize=(3, 3))

        # Left y-axis for portions
        utils.plot.quantiles(
            ax=ax,
            data=-w_0s[prompt_idx],
            color="blue",
            dim=0,
            alpha=0.1,
            label="$\\mathbf{{x}}_{{0}}$@$\\mathbf{{\\epsilon}}_{{t}}$",
        )
        utils.plot.quantiles(
            ax=ax,
            data=w_ts[prompt_idx],
            color="red",
            dim=0,
            alpha=0.1,
            label="$\\mathbf{{x}}_{{t}}$@$\\mathbf{{\\epsilon}}_{{t}}$",
        )
        ax.plot(
            range(1, len(_w_0s) + 1),
            -_w_0s,
            "--",
            color="blue",
            label="$-\\frac{{\\sqrt{{\\alpha_{{t}}}}}}{{\\sqrt{{1-\\alpha_{{t}}}}}}$",
            zorder=2,
        )
        ax.plot(
            range(1, len(_w_ts) + 1),
            _w_ts,
            "--",
            color="red",
            label="$\\frac{{1}}{{\\sqrt{{1-\\alpha_{{t}}}}}}$",
            zorder=2,
        )
        ax.set_xlabel("$t$")
        ax.set_xlim(-2.5, args.num_inference_steps + 2.5)
        ax.set_xticks([0, args.num_inference_steps], ["$T$", "$0$"])
        ax.set_ylabel("Portion@$\\mathbf{{\\epsilon}}_{{t}}$")
        ax.set_ylim(0.01, 100)
        ax.set_yscale("log", base=10)
        ax.grid(alpha=0.25)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1, labels1, loc="upper left", fontsize=10)

        plt.savefig(
            os.path.join(save_dir, f"{prompt_idx}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
