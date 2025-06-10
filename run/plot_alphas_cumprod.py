import os

import matplotlib.pyplot as plt
import utils
from paths import FIG_DIR

if __name__ == "__main__":
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas_cumprod = pipe.pipe.scheduler.alphas_cumprod

    plt.rcParams.update(
        {
            "font.size": 15,
        }
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(alphas_cumprod)
    ax.set_xlabel("Timestep")
    ax.set_xticks([0, 1000], labels=["0", "$T$"])
    ax.set_ylabel("$\\bar{\\alpha}_{t}$")
    ax.grid(alpha=0.25)
    plt.savefig(
        os.path.join(FIG_DIR, "alphas_cumprod.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close("all")
