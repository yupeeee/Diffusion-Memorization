import os
from typing import Tuple

import torch
import tqdm
import utils
from paths import DATA_DIR, LOG_DIR
from torch import Tensor


def alpha_check(
    args,
    x_ts: Tensor,
    eps_ts: Tensor,
) -> Tuple[Tensor, Tensor]:
    T = args.num_inference_steps
    device = args.device
    x_ts = x_ts.reshape(T + 1, -1).to(device)  # (T + 1, dim)
    eps_ts = eps_ts.reshape(T, -1).to(device)  # (T, dim)

    x_0s = x_ts[-1].unsqueeze(dim=0).unsqueeze(dim=0).repeat(T, T, 1)  # (T, T, dim)

    alphas = alphas_cumprod.view(-1, 1, 1).to(device)
    x_t_expanded = x_ts[:-1].unsqueeze(0)  # (1, T, dim)
    eps_expanded = eps_ts.unsqueeze(0)  # (1, T, dim)
    pred_x_0s = (x_t_expanded - (1 - alphas).sqrt() * eps_expanded) / alphas.sqrt()
    pred_x_0s = pred_x_0s.transpose(0, 1)  # (T: step, T: alpha per step, dim)

    mse = (
        utils.metric.mse(
            x_0s.reshape(T * T, -1),
            pred_x_0s.reshape(T * T, -1),
        )
        .reshape(T, T)
        .cpu()
    )

    cos = (
        utils.metric.cos(
            x_0s.reshape(T * T, -1),
            pred_x_0s.reshape(T * T, -1),
        )
        .reshape(T, T)
        .cpu()
    )

    return mse, cos


if __name__ == "__main__":
    args = utils.args.load()

    # Compute theoretical w_0s and w_Ts
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas_cumprod = pipe.alphas_cumprod(args.num_inference_steps)

    prompt_idxs = list(range(0, 10)) + list(range(500, 510))

    mses = []
    coss = []

    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Computing alpha check...",
    ):
        mses_per_prompt = []
        coss_per_prompt = []

        for seed in range(args.num_seeds):
            x_ts = torch.load(
                os.path.join(
                    LOG_DIR,
                    "sdv1-memorization",
                    f"step{args.num_inference_steps}-guid{args.guidance_scale}",
                    f"seed_{seed}",
                    "latents",
                    f"{prompt_idx}.pt",
                ),
                weights_only=True,
            )
            eps_ts_uncond, eps_ts_cond = torch.load(
                os.path.join(
                    LOG_DIR,
                    "sdv1-memorization",
                    f"step{args.num_inference_steps}-guid{args.guidance_scale}",
                    f"seed_{seed}",
                    "noise_preds",
                    f"{prompt_idx}.pt",
                ),
                weights_only=True,
            )
            eps_ts = eps_ts_uncond + args.guidance_scale * (eps_ts_cond - eps_ts_uncond)
            mse_per_prompt, cos_per_prompt = alpha_check(args, x_ts, eps_ts)
            mses_per_prompt.append(mse_per_prompt)
            coss_per_prompt.append(cos_per_prompt)

        mses_per_prompt = torch.stack(mses_per_prompt, dim=0)  # (num_seeds, T, T)
        coss_per_prompt = torch.stack(coss_per_prompt, dim=0)  # (num_seeds, T, T)
        mses.append(mses_per_prompt)
        coss.append(coss_per_prompt)

    mses = torch.stack(mses, dim=0)  # (num_prompts, num_seeds, T, T)
    coss = torch.stack(coss, dim=0)  # (num_prompts, num_seeds, T, T)

    # Save
    save_dir = os.path.join(
        DATA_DIR,
        "sdv1-memorization",
        f"step{args.num_inference_steps}-guid{args.guidance_scale}",
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(mses, os.path.join(save_dir, "alpha_check_mses.pt"))
    torch.save(coss, os.path.join(save_dir, "alpha_check_coss.pt"))
