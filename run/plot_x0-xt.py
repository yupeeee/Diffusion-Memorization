import os

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import DATA_DIR, FIG_DIR

if __name__ == "__main__":
    args = utils.args.load()
    T = 50

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

    prompt_idxs = list(range(0, 1))
    save_dir = os.path.join(
        FIG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
        "epst_decomposition_scatter",
    )
    os.makedirs(save_dir, exist_ok=True)

    for prompt_idx in tqdm.tqdm(
        # prompt_idxs,
        [0, 1, 2, 500, 501, 502],
        desc="Plotting noise prediction decompositions...",
    ):
        # Plot
        plt.rcParams["font.size"] = 15
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.plot(
            _w_ts,
            -_w_0s,
            color="red",
            linewidth=0.5,
            linestyle="--",
            marker="x",
            markersize=1,
        )

        scatter = ax.scatter(
            w_ts[prompt_idx, :, :T].flatten(),
            -w_0s[prompt_idx, :, :T].flatten(),
            s=1,
            c=torch.arange(0, T).repeat(args.num_seeds),
            cmap="viridis",
        )
        for seed in range(args.num_seeds):
            ax.plot(
                w_ts[prompt_idx, seed, :T],
                -w_0s[prompt_idx, seed, :T],
                color="black",
                linewidth=0.5,
                alpha=0.1,
            )
        cbar = plt.colorbar(scatter, label="$t$")
        cbar.set_ticks(range(0, T + 1, 10))
        cbar.set_ticklabels(range(T, -1, -10))
        ax.set_xlim(0.9, 1.5)
        ax.set_xscale("log")
        ax.set_ylim(0.01, 1.2)
        ax.set_yscale("log")
        plt.savefig(
            os.path.join(save_dir, f"{prompt_idx}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
