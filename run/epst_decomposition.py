import os
from typing import Tuple

import torch
import tqdm
import utils
from paths import DATA_DIR, LOG_DIR
from torch import Tensor


def lstsq(
    args,
    x_ts: Tensor,
    eps_ts: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    T = args.num_inference_steps
    x_ts = x_ts.reshape(T + 1, -1)  # (T + 1, dim)
    x_0s = x_ts[-1].unsqueeze(dim=0).repeat(T, 1)  # (T, dim)
    eps_ts = eps_ts.reshape(T, -1)  # (T, dim)
    x_ts = x_ts[:-1]  # (T, dim)

    # eps_t ~ w_0 * x_0 + w_t * x_t
    lstsq = torch.linalg.lstsq(
        torch.stack([x_0s, x_ts], dim=2).to(args.device),
        eps_ts[..., None].to(args.device),
    ).solution.cpu()

    w_0s, w_ts = lstsq[:, 0, 0], lstsq[:, 1, 0]  # (T,)

    eps_hat = w_0s[:, None] * x_0s[None, :] + w_ts[:, None] * x_ts[None, :]

    residuals = (eps_ts - eps_hat).reshape(T, -1)
    errs = torch.sqrt(torch.mean(residuals**2, dim=1))
    return (w_0s, w_ts, errs)


if __name__ == "__main__":
    args = utils.args.load()

    # Compute theoretical w_0s and w_ts
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )
    alphas_cumprod = pipe.alphas_cumprod(args.num_inference_steps)
    _w_0s = -alphas_cumprod.sqrt() / (1 - alphas_cumprod).sqrt()
    _w_ts = 1 / (1 - alphas_cumprod).sqrt()

    # Compute empirical w_0s and w_ts
    prompt_idxs = list(range(0, 1000))

    W_0 = []
    W_t = []
    ERR = []

    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Decomposing noise predictions...",
    ):
        w_0s_per_prompt = []
        w_ts_per_prompt = []
        errs_per_prompt = []

        for seed in range(args.num_seeds):
            # Load latents
            x_ts: Tensor = torch.load(
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
            uncond_eps_ts, cond_eps_ts = torch.load(
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
            eps_ts = uncond_eps_ts + args.guidance_scale * (cond_eps_ts - uncond_eps_ts)
            eps_ts = eps_ts.to(x_ts.dtype)

            # Decompose noise predictions
            w_0s, w_ts, errs = lstsq(args, x_ts, eps_ts)
            w_0s_per_prompt.append(w_0s)
            w_ts_per_prompt.append(w_ts)
            errs_per_prompt.append(errs)

        w_0s_per_prompt = torch.stack(w_0s_per_prompt, dim=0)
        w_ts_per_prompt = torch.stack(w_ts_per_prompt, dim=0)
        errs_per_prompt = torch.stack(errs_per_prompt, dim=0)

        W_0.append(w_0s_per_prompt)
        W_t.append(w_ts_per_prompt)
        ERR.append(errs_per_prompt)

    W_0 = torch.stack(W_0, dim=0)
    W_t = torch.stack(W_t, dim=0)
    ERR = torch.stack(ERR, dim=0)

    # Save
    torch.save(
        {
            "_w_0": _w_0s,
            "_w_t": _w_ts,
            "w_0": W_0,
            "w_t": W_t,
            "err": ERR,
        },
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "epst_decomposition.pt",
        ),
    )
