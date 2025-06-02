import os
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from .pipe import SDPipeline

__all__ = [
    "Runner",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        pipe: SDPipeline,
        save_dir: Path,
        **kwargs,
    ) -> None:
        super().__init__()
        num_inference_steps: int = kwargs.pop("num_inference_steps", 50)
        guidance_scale: float = kwargs.pop("guidance_scale", 7.5)
        seed: int = kwargs.pop("seed", 0)
        batch_size = kwargs.pop("batch_size", 1)
        num_workers = kwargs.pop("num_workers", 0)
        devices = kwargs.pop("devices", "auto")

        if devices == "auto" and torch.cuda.device_count() > 1:
            self.sync_dist = True
        else:
            self.sync_dist = (
                isinstance(devices, str)
                and "gpu:" in devices
                and len(devices.split("gpu:")[-1].split(",")) > 1
            )

        self.pipe = pipe
        self.save_dir = os.path.join(save_dir, f"seed_{seed}")

        self.config = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
        }

    def on_test_start(self) -> None:
        self.pipe.init(
            num_inference_steps=self.config["num_inference_steps"],
            guidance_scale=self.config["guidance_scale"],
        )
        self.pipe.to(self.global_rank)
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "latents"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "noise_preds"), exist_ok=True)

    def test_step(self, batch, batch_idx):
        # Unpack batch
        prompts: Tuple[str] = batch
        # Get prompt indices
        world_size = self.trainer.world_size
        rank = self.global_rank
        prompt_nos = [
            batch_idx * self.config["batch_size"] * world_size + rank + i * world_size
            for i in range(self.config["batch_size"])
        ][: len(prompts)]
        # Prepare latents
        latents = torch.randn(
            size=(
                1,
                self.pipe.pipe.unet.config.in_channels,
                self.pipe.pipe.unet.config.sample_size,
                self.pipe.pipe.unet.config.sample_size,
            ),
            generator=torch.Generator().manual_seed(self.config["seed"]),
        )
        latents = latents.repeat(len(prompts), 1, 1, 1)
        # Generate
        images, latents, noise_preds = self.pipe(
            prompt=list(prompts),
            latents=latents,
        )
        # Save
        for prompt_no, image, latent, noise_pred in zip(
            prompt_nos, images, latents, noise_preds
        ):
            image.save(os.path.join(self.save_dir, "images", f"{prompt_no}.png"))
            torch.save(
                latent, os.path.join(self.save_dir, "latents", f"{prompt_no}.pt")
            )
            torch.save(
                noise_pred,
                os.path.join(self.save_dir, "noise_preds", f"{prompt_no}.pt"),
            )
        del images, latents, noise_preds
        torch.cuda.empty_cache()

    def on_test_end(self) -> None:
        pass


def load_accelerator_and_devices(
    devices: str = "auto",
) -> Tuple[str, List[int]]:
    if isinstance(devices, str) and "cuda" in devices:
        devices = devices.replace("cuda", "gpu")
    devices_cfg: List[str] = devices.split(":")
    accelerator = "auto"
    devices = "auto"
    if len(devices_cfg) == 1:
        accelerator = devices_cfg[0]
    else:
        accelerator = devices_cfg[0]
        devices = [int(d) for d in devices_cfg[1].split(",")]
    return accelerator, devices


def load_runner(
    **kwargs,
) -> L.Trainer:
    # Load accelerator and devices
    accelerator, devices = load_accelerator_and_devices(kwargs["devices"])
    # Set precision
    torch.set_float32_matmul_precision("medium")
    # Load runner
    runner = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed",
        logger=False,
        callbacks=None,
        max_epochs=1,
        log_every_n_steps=None,
        enable_checkpointing=False,
        deterministic=False,
    )
    # Return runner
    return runner


class Runner:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        - `**kwargs`: Configuration parameters
        """
        self.runner: L.Trainer = load_runner(**kwargs)
        self.config = kwargs

    def run(
        self,
        dataset,
        pipe: SDPipeline,
        save_dir: Path,
    ) -> None:
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
        os.makedirs(save_dir, exist_ok=True)
        model: Wrapper = Wrapper(pipe, save_dir, **self.config)
        self.runner.test(
            model,
            dataloader,
        )
