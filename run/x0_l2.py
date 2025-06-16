import os

import torch
import tqdm
import utils
from paths import DATA_DIR, LOG_DIR

if __name__ == "__main__":
    args = utils.args.load()

    x_0_l2s = []

    # Compute L2 distances between different x0s
    for prompt_idx in tqdm.trange(
        1000,
        desc="Computing ||x_0 - x_0'||_2...",
    ):
        x_0s = (
            torch.stack(
                [
                    torch.load(
                        os.path.join(
                            LOG_DIR,
                            "sdv1-memorization",
                            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
                            f"seed_{seed}",
                            "latents",
                            f"{prompt_idx}.pt",
                        ),
                        weights_only=True,
                    )[-1]
                    for seed in range(args.num_seeds)
                ],
                dim=0,
            )
            .reshape(args.num_seeds, -1)
            .to(args.device)
        )  # (num_seeds, latent_dim)

        diffs = x_0s.unsqueeze(0) - x_0s.unsqueeze(
            1
        )  # (num_seeds, num_seeds, latent_dim)
        l2s = diffs.norm(p=2, dim=-1)  # (num_seeds, num_seeds)
        x_0_l2s.append(l2s.cpu())

    x_0_l2s = torch.stack(x_0_l2s, dim=0)  # (num_prompts, num_seeds, num_seeds)

    # Save
    save_dir = os.path.join(
        DATA_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(x_0_l2s, os.path.join(save_dir, f"x0_l2s.pt"))
