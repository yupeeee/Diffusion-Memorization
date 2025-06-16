import os

import matplotlib.pyplot as plt
import tqdm
import utils
from paths import FIG_DIR, LOG_DIR
from PIL import Image

if __name__ == "__main__":
    args = utils.args.load()
    num_seeds = 10
    prompt_idxs = list(range(0, 14))  # + list(range(500, 510))

    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Plotting generation examples...",
    ):
        # Load generated images
        images = [
            Image.open(
                os.path.join(
                    LOG_DIR,
                    "sdv1-memorization",
                    f"step{args.num_inference_steps}-guid{args.guidance_scale}",
                    f"seed_{seed}",
                    "images",
                    f"{prompt_idx}.png",
                )
            )
            for seed in range(num_seeds)
        ]
        # Plot
        fig, ax = plt.subplots(1, num_seeds, figsize=(num_seeds, 1))
        plt.subplots_adjust(wspace=0)  # Remove horizontal space between subplots
        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].axis("off")
        save_dir = os.path.join(
            FIG_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "examples",
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{prompt_idx}.pdf"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
