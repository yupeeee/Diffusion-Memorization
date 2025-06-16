import os

import torch
import tqdm
import utils
from paths import DATA_DIR, LOG_DIR, SSCD_WEIGHTS_PATH
from PIL import Image

if __name__ == "__main__":
    args = utils.args.load()

    sscd = utils.sscd.SSCD(
        model_path=SSCD_WEIGHTS_PATH,
        device=args.device,
    )

    embeddings = []

    # Compute SSCD embeddings
    for prompt_idx in tqdm.trange(
        1000,
        desc="Computing SSCD embeddings...",
    ):
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
            for seed in range(args.num_seeds)
        ]
        embeddings.append(sscd(images))

    embeddings = torch.stack(embeddings, dim=0)  # (num_prompts, num_seeds, 512)

    # Save
    save_dir = os.path.join(
        DATA_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        embeddings,
        os.path.join(
            save_dir,
            f"sscd_embeddings_{args.num_seeds}seeds.pt",
        ),
    )
