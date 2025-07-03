import os

import matplotlib.pyplot as plt
import torch
import tqdm
import utils
from paths import DATA_DIR, FIG_DIR


if __name__ == "__main__":
    args = utils.args.load()

    # Load data
    data_dir = os.path.join(
        DATA_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    mses = torch.load(os.path.join(data_dir, f"pred_x0_mses.pt"), weights_only=True)
    coss = torch.load(os.path.join(data_dir, f"pred_x0_coss.pt"), weights_only=True)

    mses_mem, mses_nor = mses[:500], mses[500:]
    coss_mem, coss_nor = coss[:500], coss[500:]

    # Plot
    save_dir = os.path.join(
        FIG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
        "pred_x0_err",
    )
    os.makedirs(os.path.join(save_dir, "mse"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "cos"), exist_ok=True)

    plt.rcParams["font.size"] = 15

    # Aggregate (MSE)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    utils.plot.quantiles(ax, mses_mem.reshape(-1, args.num_inference_steps), "red")
    utils.plot.quantiles(ax, mses_nor.reshape(-1, args.num_inference_steps), "blue")
    ax.set_xlabel("$t$")
    ax.set_xlim(-2.25, args.num_inference_steps + 2.25)
    ax.set_xticks([0, 10, 20, 30, 40, 50], labels=["50", "40", "30", "20", "10", "0"])
    ax.set_ylabel(
        "$\\mathrm{{MSE}}(\\hat{{\\mathbf{{x}}}}_0^{{(t)}}, \\mathbf{{x}}_0)$"
    )
    # ax.set_ylim(-0.05, 1.05)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", ".2", ".4", ".6", ".8", "1"])
    ax.grid(alpha=0.25)
    plt.savefig(
        os.path.join(save_dir, f"mse.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    # Aggregate (Cosine similarity)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    utils.plot.quantiles(ax, coss_mem.reshape(-1, args.num_inference_steps), "red")
    utils.plot.quantiles(ax, coss_nor.reshape(-1, args.num_inference_steps), "blue")
    ax.set_xlabel("$t$")
    ax.set_xlim(-2.25, args.num_inference_steps + 2.25)
    ax.set_xticks([0, 10, 20, 30, 40, 50], labels=["50", "40", "30", "20", "10", "0"])
    ax.set_ylabel(
        "$\\mathrm{{Cos}}(\\hat{{\\mathbf{{x}}}}_0^{{(t)}}, \\mathbf{{x}}_0)$"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", ".2", ".4", ".6", ".8", "1"]
    )
    ax.grid(alpha=0.25)
    plt.savefig(
        os.path.join(save_dir, f"cos.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    for i, (mse_mem, mse_nor) in tqdm.tqdm(
        enumerate(zip(mses_mem, mses_nor)),
        desc="Plotting MSEs...",
        total=len(mses_mem),
    ):
        # Memorized
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        utils.plot.quantiles(ax, mse_mem, "red")
        ax.set_xlabel("$t$")
        ax.set_xlim(-2.25, args.num_inference_steps + 2.25)
        ax.set_xticks(
            [0, 10, 20, 30, 40, 50], labels=["50", "40", "30", "20", "10", "0"]
        )
        ax.set_ylabel(
            "$\\mathrm{{MSE}}(\\hat{{\\mathbf{{x}}}}_0^{{(t)}}, \\mathbf{{x}}_0)$"
        )
        # ax.set_ylim(-0.05, 1.05)
        # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", ".2", ".4", ".6", ".8", "1"])
        ax.grid(alpha=0.25)
        plt.savefig(
            os.path.join(save_dir, "mse", f"{i}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)

        # Normal
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        utils.plot.quantiles(ax, mse_nor, "blue")
        ax.set_xlabel("$t$")
        ax.set_xlim(-2.25, args.num_inference_steps + 2.25)
        ax.set_xticks(
            [0, 10, 20, 30, 40, 50], labels=["50", "40", "30", "20", "10", "0"]
        )
        ax.set_ylabel(
            "$\\mathrm{{MSE}}(\\hat{{\\mathbf{{x}}}}_0^{{(t)}}, \\mathbf{{x}}_0)$"
        )
        # ax.set_ylim(-0.05, 1.05)
        # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", ".2", ".4", ".6", ".8", "1"])
        ax.grid(alpha=0.25)
        plt.savefig(
            os.path.join(save_dir, "mse", f"{500 + i}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)

    for i, (cos_mem, cos_nor) in tqdm.tqdm(
        enumerate(zip(coss_mem, coss_nor)),
        desc="Plotting Cosine similarities...",
        total=len(coss_mem),
    ):
        # Memorized
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        utils.plot.quantiles(ax, cos_mem, "red")
        ax.set_xlabel("$t$")
        ax.set_xlim(-2.25, args.num_inference_steps + 2.25)
        ax.set_xticks(
            [0, 10, 20, 30, 40, 50], labels=["50", "40", "30", "20", "10", "0"]
        )
        ax.set_ylabel(
            "$\\mathrm{{Cos}}(\\hat{{\\mathbf{{x}}}}_0^{{(t)}}, \\mathbf{{x}}_0)$"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(
            [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", ".2", ".4", ".6", ".8", "1"]
        )
        ax.grid(alpha=0.25)
        plt.savefig(
            os.path.join(save_dir, "cos", f"{i}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)

        # Normal
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        utils.plot.quantiles(ax, cos_nor, "blue")
        ax.set_xlabel("$t$")
        ax.set_xlim(-2.25, args.num_inference_steps + 2.25)
        ax.set_xticks(
            [0, 10, 20, 30, 40, 50], labels=["50", "40", "30", "20", "10", "0"]
        )
        ax.set_ylabel(
            "$\\mathrm{{Cos}}(\\hat{{\\mathbf{{x}}}}_0^{{(t)}}, \\mathbf{{x}}_0)$"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(
            [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", ".2", ".4", ".6", ".8", "1"]
        )
        ax.grid(alpha=0.25)
        plt.savefig(
            os.path.join(save_dir, "cos", f"{500 + i}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
