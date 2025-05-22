import argparse

import matplotlib.pyplot as plt
import torch
import utils
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--cuda", type=int, required=False, default=0)
    args = parser.parse_args()

    # Load pipeline
    pipe = (
        utils.pipe.StableDiffusion(
            version="1.4",
            scheduler="DDIM",
            variant="fp16",
            verbose=True,
        )
        .to(args.cuda)
        .seed(args.seed)
    )

    # Generate images
    images, latents, noise_preds = pipe(args.prompt)

    # Calculate pred_x0s
    alphas = pipe.alphas()
    pred_x0s = []
    epss = []
    xts = []
    for alpha, latent, noise_pred, xt in zip(
        alphas, latents[:-1], noise_preds, latents[1:]
    ):
        noise_pred = pipe.classifier_free_guidance(noise_pred)
        pred_x0 = (latent - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
        pred_x0, eps, xt = pipe.decode(torch.cat([pred_x0, noise_pred, xt], dim=0))
        pred_x0s.append(pred_x0)
        epss.append(eps)
        xts.append(xt)

    # Save images
    fig, axes = plt.subplots(10, 5, figsize=(15, 10))
    axes = axes.flatten()

    for seed, (pred_x0, eps, xt) in enumerate(zip(pred_x0s, epss, xts)):
        # Concatenate pred_x0 and eps horizontally
        image = Image.new("RGB", (pred_x0.width + eps.width + xt.width, pred_x0.height))
        image.paste(pred_x0, (0, 0))
        image.paste(eps, (pred_x0.width, 0))
        image.paste(xt, (pred_x0.width + eps.width, 0))
        axes[seed].imshow(image)
        axes[seed].axis("off")
    filename = "".join(c for c in args.prompt if c.isalnum() or c in (" ", "-", "_"))
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
