import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import DATA_DIR, FIG_DIR
from torch import Tensor

if __name__ == "__main__":
    args = utils.args.load()

    data_dir = os.path.join(
        DATA_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    mses = torch.load(os.path.join(data_dir, "alpha_check_mses.pt"), weights_only=False)
    coss = torch.load(os.path.join(data_dir, "alpha_check_coss.pt"), weights_only=False)

    save_dir = os.path.join(
        FIG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
        "alpha_check",
    )
    os.makedirs(os.path.join(save_dir, "mses"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "coss"), exist_ok=True)

    for i, (mses_per_prompt, coss_per_prompt) in tqdm.tqdm(
        enumerate(zip(mses, coss)),
        desc="Plotting alpha check...",
        total=len(mses),
    ):
        fig, axes = plt.subplots(5, 10, figsize=(30, 15))
        for ax, mse in zip(axes.flatten(), mses_per_prompt):
            im = ax.matshow(mse.log())
            plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(save_dir, "mses", f"{i}.png"))
        plt.close(fig)

        fig, axes = plt.subplots(5, 10, figsize=(30, 15))
        for ax, cos in zip(axes.flatten(), coss_per_prompt):
            im = ax.matshow(cos, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
        plt.savefig(
            os.path.join(save_dir, "coss", f"{i}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
