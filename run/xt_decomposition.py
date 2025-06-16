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
) -> Tuple[Tensor, Tensor, Tensor]:
    T = args.num_inference_steps
    x_ts = x_ts.reshape(T + 1, -1)  # (T + 1, dim)
    x_T = x_ts[0].unsqueeze(dim=0).repeat(T, 1)  # (T, dim)
    x_0 = x_ts[-1].unsqueeze(dim=0).repeat(T, 1)  # (T, dim)
    X = x_ts[1:]  # (T, dim)

    # X ~ w_0 * x_0 + w_T * x_T
    lstsq = torch.linalg.lstsq(
        torch.stack([x_0, x_T], dim=2),
        X[..., None],
    ).solution

    w_0s, w_Ts = lstsq[:, 0, 0], lstsq[:, 1, 0]  # (T,)

    X_hat = w_0s[:, None] * x_0[None, :] + w_Ts[:, None] * x_T[None, :]

    residuals = (X - X_hat).reshape(T, -1)
    errs = torch.sqrt(torch.mean(residuals**2, dim=1))
    return (w_0s, w_Ts, errs)


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
    _w_0s = alphas_cumprod.sqrt()
    _w_Ts = (1 - alphas_cumprod).sqrt()

    # Compute empirical w_0s and w_Ts
    prompt_idxs = list(range(0, 1000))

    W_0 = []
    W_T = []
    ERR = []

    for prompt_idx in tqdm.tqdm(
        prompt_idxs,
        desc="Decomposing latents...",
    ):
        w_0s_per_prompt = []
        w_Ts_per_prompt = []
        errs_per_prompt = []

        for seed in range(args.num_seeds):
            # Load latents
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

            # Decompose latents
            w_0s, w_Ts, errs = lstsq(args, x_ts)
            w_0s_per_prompt.append(w_0s)
            w_Ts_per_prompt.append(w_Ts)
            errs_per_prompt.append(errs)

        w_0s_per_prompt = torch.stack(w_0s_per_prompt, dim=0)
        w_Ts_per_prompt = torch.stack(w_Ts_per_prompt, dim=0)
        errs_per_prompt = torch.stack(errs_per_prompt, dim=0)

        W_0.append(w_0s_per_prompt)
        W_T.append(w_Ts_per_prompt)
        ERR.append(errs_per_prompt)

    W_0 = torch.stack(W_0, dim=0)
    W_T = torch.stack(W_T, dim=0)
    ERR = torch.stack(ERR, dim=0)

    # Save
    torch.save(
        {
            "_w_0": _w_0s,
            "_w_T": _w_Ts,
            "w_0": W_0,
            "w_T": W_T,
            "err": ERR,
        },
        os.path.join(
            DATA_DIR,
            "sdv1-memorization",
            f"step{args.num_inference_steps}-guid{args.guidance_scale}",
            "xt_decomposition.pt",
        ),
    )
