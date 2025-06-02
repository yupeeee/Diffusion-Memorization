import os
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from .pipe import SDPipeline

__all__ = [
    "Trainer",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        pipe: SDPipeline,
        **kwargs,
    ) -> None:
        super().__init__()
        num_inference_steps: int = kwargs.pop("num_inference_steps", 50)
        guidance_scale: float = kwargs.pop("guidance_scale", 7.5)
        p_uncond: float = kwargs.pop("p_uncond", 0.1)
        batch_size = kwargs.pop("batch_size", 1)
        num_workers = kwargs.pop("num_workers", 0)
        num_epochs = kwargs.pop("num_epochs", 1)
        lr = kwargs.pop("lr", 1e-4)
        lr_warmup_steps = kwargs.pop("lr_warmup_steps", 0)
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
        self.unet = self.pipe.pipe.unet
        self.criterion = torch.nn.functional.mse_loss
        self.lr = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.config = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "p_uncond": p_uncond,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "num_epochs": num_epochs,
            "lr": lr,
            "lr_warmup_steps": lr_warmup_steps,
            "devices": devices,
        }
        self.save_hyperparameters(self.config)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.config["lr"])
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config["lr_warmup_steps"],
            num_training_steps=self.config["num_epochs"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def on_train_epoch_start(self) -> None:
        self.pipe.init(
            num_inference_steps=1000,
            guidance_scale=1.0,
        )
        self.pipe.to(self.global_rank)

    def on_validation_epoch_start(self) -> None:
        self.pipe.init(
            num_inference_steps=self.config["num_inference_steps"],
            guidance_scale=self.config["guidance_scale"],
        )
        self.pipe.to(self.global_rank)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        images: Tensor
        prompts: Tuple[str]
        images, prompts = batch
        # Prepare
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.config["num_inference_steps"], (batch_size,)
        ).long()
        # Apply unconditional guidance with probability p_uncond
        prompts = [
            "" if torch.rand(1).item() < self.config["p_uncond"] else p for p in prompts
        ]
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=list(prompts),
        )
        # Forward pass
        noise = torch.randn_like(images)
        noisy_images = self.pipe.add_noise(images, noise, timesteps)
        # Predict noise
        noise_pred = self.pipe.predict_noise(
            noisy_samples=noisy_images.to(self.global_rank),
            ts=timesteps.to(self.global_rank),
            prompt_embeds=prompt_embeds.to(self.global_rank),
        )
        # Compute loss
        loss: Tensor = self.criterion(noise, noise_pred)
        self.train_loss(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1
        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)
        # log: lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        lr = scheduler.get_last_lr()[0]
        self.lr(lr)
        self.log("lr", self.lr, sync_dist=self.sync_dist)
        # log: loss, acc@1
        self.log("train/avg_loss", self.train_loss, sync_dist=self.sync_dist)
        # Save checkpoint
        if self.trainer.is_global_zero:
            if "save_per_epoch" in self.config.keys():
                if (current_epoch + 1) % self.config["save_per_epoch"] == 0:
                    ckpt_dir = os.path.join(self.trainer._log_dir, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        self.pipe.pipe.unet.state_dict(),
                        os.path.join(ckpt_dir, f"epoch={current_epoch}.ckpt"),
                    )

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        images: Tensor
        prompts: Tuple[str]
        images, prompts = batch
        # Prepare
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.config["num_inference_steps"], (batch_size,)
        ).long()
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=list(prompts),
        )
        # Forward pass
        noise = torch.randn_like(images)
        noisy_images = self.pipe.add_noise(images, noise, timesteps)
        # Predict noise
        noise_pred = self.pipe.predict_noise(
            noisy_samples=noisy_images.to(self.global_rank),
            ts=timesteps.to(self.global_rank),
            prompt_embeds=prompt_embeds.to(self.global_rank),
        )
        if self.pipe.pipe.do_classifier_free_guidance:
            noise_pred = self.pipe.classifier_free_guidance(noise_pred)
        # Compute loss
        loss: Tensor = self.criterion(noise, noise_pred)
        self.val_loss(loss)

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1
        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)
        # log: loss
        self.log("val/avg_loss", self.val_loss, sync_dist=self.sync_dist)


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


def load_checkpoint_callback() -> ModelCheckpoint:
    return ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        save_on_train_epoch_end=True,
    )


def load_tensorboard_logger(
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
    )


def load_trainer(
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
    **kwargs,
) -> L.Trainer:
    # Load accelerator and devices
    accelerator, devices = load_accelerator_and_devices(kwargs["devices"])
    # Load tensorboard logger
    tensorboard_logger = load_tensorboard_logger(
        save_dir=save_dir,
        name=name,
        version=version,
    )
    _log_dir = tensorboard_logger.log_dir
    checkpoint_callback = load_checkpoint_callback()
    # Set precision
    torch.set_float32_matmul_precision("medium")
    # Load trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed",
        logger=[
            tensorboard_logger,
        ],
        callbacks=[
            checkpoint_callback,
        ],
        max_epochs=kwargs["num_epochs"],
        log_every_n_steps=None,
        deterministic=False,
    )
    # Set log directory
    setattr(trainer, "_log_dir", _log_dir)
    # Return trainer
    return trainer


class Trainer:
    def __init__(
        self,
        log_root: Path,
        name: str,
        version: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        - `log_root` (Path): Root directory for saving logs
        - `name` (str): Name of the experiment
        - `version` (Optional[str]): Version of the experiment
        - `**kwargs`: Configuration parameters
        """
        self.trainer: L.Trainer = load_trainer(log_root, name, version, **kwargs)
        self.log_dir = self.trainer._log_dir
        self.config = kwargs

    def run(
        self,
        train_dataset,
        val_dataset,
        pipe: SDPipeline,
        resume: bool = False,
    ) -> None:
        train_dataloader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        val_dataloader: DataLoader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
        model: Wrapper = Wrapper(pipe, **self.config)
        self.trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path="last" if resume else None,
        )
