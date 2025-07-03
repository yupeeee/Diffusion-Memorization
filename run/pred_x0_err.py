import os

import torch
import tqdm
import utils
from paths import DATA_DIR, LOG_DIR


def mse(x, y):
    return ((x - y) ** 2).mean(dim=-1)


def cos(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


if __name__ == "__main__":
    args = utils.args.load()

    # Load alphas_cumprod
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas_cumprod = pipe.alphas_cumprod(args.num_inference_steps)

    # Reshape alphas for broadcasting: (T, 1, 1, 1)
    alphas_cumprod = alphas_cumprod.view(-1, 1, 1, 1).to(args.device)

    log_dir = os.path.join(
        LOG_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    save_dir = os.path.join(
        DATA_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    os.makedirs(save_dir, exist_ok=True)

    mses = []
    coss = []

    prompt_idxs = list(range(0, 1000))

    for i in tqdm.tqdm(
        prompt_idxs,
        desc=f"Calculating MSE/Cosine x_0 estimation errors...",
    ):
        mses_per_prompt = []
        coss_per_prompt = []

        for seed in range(args.num_seeds):
            x_ts = torch.load(
                os.path.join(
                    log_dir,
                    f"seed_{seed}",
                    "latents",
                    f"{i}.pt",
                ),
                weights_only=True,
            )
            eps_ts_uncond, eps_ts_cond = torch.load(
                os.path.join(
                    log_dir,
                    f"seed_{seed}",
                    "noise_preds",
                    f"{i}.pt",
                ),
                weights_only=True,
            )
            eps_ts = eps_ts_uncond + args.guidance_scale * (eps_ts_cond - eps_ts_uncond)
            x_ts = x_ts.to(args.device)
            eps_ts = eps_ts.to(args.device)

            x_0s = x_ts[-1].unsqueeze(dim=0).repeat(args.num_inference_steps, 1, 1, 1)
            pred_x_0s = (
                x_ts[:-1] - (1 - alphas_cumprod).sqrt() * eps_ts
            ) / alphas_cumprod.sqrt()

            x_0s = x_0s.reshape(args.num_inference_steps, -1)
            pred_x_0s = pred_x_0s.reshape(args.num_inference_steps, -1)

            mses_per_prompt.append(mse(x_0s, pred_x_0s).cpu())
            coss_per_prompt.append(cos(x_0s, pred_x_0s).cpu())

        mses.append(torch.stack(mses_per_prompt, dim=0))
        coss.append(torch.stack(coss_per_prompt, dim=0))

    mses = torch.stack(mses, dim=0)
    coss = torch.stack(coss, dim=0)

    torch.save(mses, os.path.join(save_dir, f"pred_x0_mses.pt"))
    torch.save(coss, os.path.join(save_dir, f"pred_x0_coss.pt"))
