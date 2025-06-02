import argparse
import os

import torch
import tqdm
import utils
from paths import DATASET_DIR, LOG_DIR, DATA_DIR


def mse(x, y):
    return ((x - y) ** 2).mean(dim=-1)


def cos(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-inference-steps", type=int, required=False, default=50)
    parser.add_argument("--guidance-scale", type=float, required=False, default=7.5)
    parser.add_argument("--seed", type=int, required=False, default=0)
    args = parser.parse_args()

    # Load dataset
    prompts = utils.datasets.sdv1_memorization.Prompts(
        root=os.path.join(DATASET_DIR, "sdv1-memorization"),
    )

    # Load alphas_cumprod
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas_cumprod = pipe.alphas_cumprod(args.num_inference_steps)
    alphas_cumprod = torch.cat(
        [alphas_cumprod, torch.ones_like(alphas_cumprod[-1:])], dim=0
    )
    # Reshape alphas for broadcasting: (T + 1, 1, 1, 1)
    alphas_cumprod = alphas_cumprod.view(-1, 1, 1, 1)

    log_dir = os.path.join(LOG_DIR, "sdv1-memorization", f"seed_{args.seed}")
    save_dir = os.path.join(DATA_DIR, "sdv1-memorization", f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    mses = {
        "x_0": [],
        "eps": [],
        "eps_uncond": [],
        "eps_cond": [],
    }
    coss = {
        "x_0": [],
        "eps": [],
        "eps_uncond": [],
        "eps_cond": [],
    }

    for i in tqdm.trange(
        len(prompts),
        desc=f"Calculating errors (seed {args.seed})",
    ):
        x_ts = torch.load(
            os.path.join(log_dir, "latents", f"{i}.pt"), weights_only=True
        )
        eps_ts_uncond, eps_ts_cond = torch.load(
            os.path.join(log_dir, "noise_preds", f"{i}.pt"), weights_only=True
        )
        eps_ts = eps_ts_uncond + args.guidance_scale * (eps_ts_cond - eps_ts_uncond)

        x_0s = x_ts[-1].unsqueeze(dim=0).repeat(args.num_inference_steps, 1, 1, 1)
        epss = (x_ts[:-1] - alphas_cumprod[:-1].sqrt() * x_0s) / (
            1 - alphas_cumprod[:-1]
        ).sqrt()
        x_0_preds = (
            x_ts[:-1] - (1 - alphas_cumprod[:-1]).sqrt() * eps_ts
        ) / alphas_cumprod[:-1].sqrt()
        x_0_preds_next = (
            x_ts[:-1] - (1 - alphas_cumprod[1:]).sqrt() * eps_ts
        ) / alphas_cumprod[1:].sqrt()

        x_0s = x_0s.reshape(args.num_inference_steps, -1)
        x_0_preds = x_0_preds.reshape(args.num_inference_steps, -1)
        epss = epss.reshape(args.num_inference_steps, -1)
        eps_ts = eps_ts.reshape(args.num_inference_steps, -1)
        eps_ts_uncond = eps_ts_uncond.reshape(args.num_inference_steps, -1)
        eps_ts_cond = eps_ts_cond.reshape(args.num_inference_steps, -1)

        mses["x_0"].append(mse(x_0s, x_0_preds))
        mses["eps"].append(mse(epss, eps_ts))
        mses["eps_uncond"].append(mse(epss, eps_ts_uncond))
        mses["eps_cond"].append(mse(epss, eps_ts_cond))

        coss["x_0"].append(cos(x_0s, x_0_preds))
        coss["eps"].append(cos(epss, eps_ts))
        coss["eps_uncond"].append(cos(epss, eps_ts_uncond))
        coss["eps_cond"].append(cos(epss, eps_ts_cond))

    mses = {k: torch.stack(v, dim=0) for k, v in mses.items()}
    coss = {k: torch.stack(v, dim=0) for k, v in coss.items()}

    for k in mses.keys():
        torch.save(mses[k], os.path.join(save_dir, f"mses_{k}.pt"))
        torch.save(coss[k], os.path.join(save_dir, f"coss_{k}.pt"))
