import argparse
import os
from argparse import Namespace
from typing import Tuple

import torch
import tqdm
import utils
from paths import DATA_DIR, DATASET_DIR, LOG_DIR
from torch import Tensor


def lsq(
    args: Namespace,
    x_ts: Tensor,
    pred_x_0s: Tensor,
) -> Tuple[
    Tuple[Tensor, Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor],
]:
    num_inference_steps = args.num_inference_steps
    x_ts = x_ts.reshape(num_inference_steps + 1, -1)  # (T + 1, dim)
    pred_x_0s = pred_x_0s.reshape(num_inference_steps, -1)  # (T, dim)
    x_T = x_ts[0].unsqueeze(dim=0).repeat(num_inference_steps, 1)  # (T, dim)
    x_0 = x_ts[-1].unsqueeze(dim=0).repeat(num_inference_steps, 1)  # (T, dim)
    X = x_ts[:-1]  # (T, dim)

    # X ~ a * x_T + b * x_0
    lsq_0 = torch.linalg.lstsq(
        torch.stack([x_T, x_0], dim=2),
        X[..., None],
    ).solution
    lsq_t = torch.linalg.lstsq(
        torch.stack([x_T, pred_x_0s], dim=2),
        X[..., None],
    ).solution

    a_0s, b_0s = lsq_0[:, 0, 0], lsq_0[:, 1, 0]  # (T,)
    a_ts, b_ts = lsq_t[:, 0, 0], lsq_t[:, 1, 0]  # (T,)

    X_hat_0 = a_0s[:, None] * x_T[None, :] + b_0s[:, None] * x_0[None, :]
    X_hat_t = a_ts[:, None] * x_T[None, :] + b_ts[:, None] * pred_x_0s[None, :]

    residuals_0 = X - X_hat_0
    residuals_t = X - X_hat_t
    errs_0 = torch.norm(residuals_0, dim=1)
    errs_t = torch.norm(residuals_t, dim=1)

    return (a_0s, b_0s, errs_0), (a_ts, b_ts, errs_t)


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

    data_0 = {
        "x_T@x_t": [],
        "x_0@x_t": [],
        "err": [],
    }
    data_t = {
        "x_T@x_t": [],
        "x_0@x_t": [],
        "err": [],
    }

    for i in tqdm.trange(
        len(prompts),
        desc=f"LSQ fitting (seed {args.seed})",
    ):
        # Load data
        x_ts = torch.load(
            os.path.join(log_dir, "latents", f"{i}.pt"), weights_only=True
        )
        eps_ts_uncond, eps_ts_cond = torch.load(
            os.path.join(log_dir, "noise_preds", f"{i}.pt"), weights_only=True
        )
        eps_ts = eps_ts_uncond + args.guidance_scale * (eps_ts_cond - eps_ts_uncond)
        x_0_preds = (
            x_ts[:-1] - (1 - alphas_cumprod[:-1]).sqrt() * eps_ts
        ) / alphas_cumprod[:-1].sqrt()

        # LSQ fitting
        (a_0s, b_0s, errs_0), (a_ts, b_ts, errs_t) = lsq(args, x_ts, x_0_preds)
        data_0["x_T@x_t"].append(a_0s)
        data_0["x_0@x_t"].append(b_0s)
        data_0["err"].append(errs_0)
        data_t["x_T@x_t"].append(a_ts)
        data_t["x_0@x_t"].append(b_ts)
        data_t["err"].append(errs_t)

    data_0 = {k: torch.stack(v, dim=0) for k, v in data_0.items()}
    data_t = {k: torch.stack(v, dim=0) for k, v in data_t.items()}
    torch.save(data_0, os.path.join(save_dir, f"lsq_0.pt"))
    torch.save(data_t, os.path.join(save_dir, f"lsq_t.pt"))
