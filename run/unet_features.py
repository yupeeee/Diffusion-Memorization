import os

import torch
import tqdm
import utils
from paths import DATASET_DIR, LOG_DIR

if __name__ == "__main__":
    args = utils.args.load()
    prompt_idxs = list(range(0, 1000))

    # Load dataset
    prompts = utils.datasets.sdv1_memorization.Prompts(
        root=os.path.join(DATASET_DIR, "sdv1-memorization"),
    )

    # Load pipeline
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    pipe.init(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    pipe.to(args.device)

    # Extract features
    unet_extractor = utils.unet.UNetExtractor(
        pipe=pipe,
    )
    save_dir = os.path.join(
        LOG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
        f"seed_{args.seed}",
        "unet_features@T",
    )
    os.makedirs(save_dir, exist_ok=True)
    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Extracting UNet features...",
    ):
        unet_features = unet_extractor(
            prompt=prompts[prompt_idx],
            seed=args.seed,
            start_step=0,
            exit_step=1,
        )
        torch.save(
            unet_features,
            os.path.join(save_dir, f"{prompt_idx}.pt"),
        )
