import os

import matplotlib.pyplot as plt
import torch
import utils
from paths import DATA_DIR, FIG_DIR

if __name__ == "__main__":
    args = utils.args.load()

    # Load data
    x0_l2s = torch.load(
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "x0_l2s.pt",
        ),
        weights_only=True,
    )
    sscd_embeddings = torch.load(
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "sscd_embeddings.pt",
        ),
        weights_only=True,
    )
    sscd_scores = torch.nn.functional.cosine_similarity(
        sscd_embeddings.unsqueeze(1),  # (1000, 1, 50, 512)
        sscd_embeddings.unsqueeze(2),  # (1000, 50, 1, 512)
        dim=-1,
    )  # (1000, 50, 50)

    # Plot
    x0_l2s_mem = x0_l2s[:500].mean(dim=-1)
    sscd_scores_mem = sscd_scores[:500].mean(dim=-1)
    x0_l2s_nor = x0_l2s[500:].mean(dim=-1)
    sscd_scores_nor = sscd_scores[500:].mean(dim=-1)

    plt.rcParams["font.size"] = 15
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(x0_l2s_nor, sscd_scores_nor, c="blue", s=1, alpha=0.01)
    ax.scatter(x0_l2s_mem, sscd_scores_mem, c="red", s=1, alpha=0.01)
    ax.set_xlabel("$\\mathbb{{E}}[l_{{2}}(\\mathbf{{x}}_{{0}}, \\mathbf{{x}}_{{0}}')]$")
    ax.set_xlim(-10, 210)
    ax.set_xticks([0, 100, 200], labels=["0", "100", "200"])
    ax.set_ylabel("SSCD score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", ".25", ".50", ".75", "1"])
    ax.grid(alpha=0.25)
    save_dir = os.path.join(
        FIG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
        "sscd-l2",
    )
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "sscd-l2.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
