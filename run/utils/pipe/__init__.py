from pathlib import Path
from typing import Optional, Union

import torch

from . import v1, v1_cifar10, v2
from ._unet_layers import unet_layers

__all__ = [
    "SDPipeline",
    "unet_layers",
    "StableDiffusion",
    "StableDiffusionCIFAR10",
]

SDPipeline = Union[v1.StableDiffusionV1, v2.StableDiffusionV2]


def emph(text: str) -> str:
    return f"\033[1m\033[3m{text}\033[0m"


def StableDiffusion(
    version: str,
    scheduler: str,
    variant: str = "fp16",
    verbose: bool = False,
) -> Union[v1.StableDiffusionV1, v2.StableDiffusionV2]:
    if version in v1.model_path.keys():
        pipe = v1.StableDiffusionV1
    elif version in v2.model_path.keys():
        pipe = v2.StableDiffusionV2
    else:
        supported_versions = list(v1.model_path.keys()) + list(v2.model_path.keys())
        supported_versions = [emph(v) for v in supported_versions]
        supported_versions = ", ".join(supported_versions)
        raise ValueError(
            f"Unsupported version {emph(version)} (Supported versions: {supported_versions})"
        )
    return pipe(
        version=version,
        scheduler=scheduler,
        variant=variant,
        verbose=verbose,
    )


def StableDiffusionCIFAR10(
    version: str,
    scheduler: str,
    variant: str = "fp16",
    verbose: bool = False,
    weights_path: Optional[Path] = None,
) -> v1_cifar10.StableDiffusionV1:
    pipe = v1_cifar10.StableDiffusionV1(
        version=version,
        scheduler=scheduler,
        variant=variant,
        verbose=verbose,
    )
    if weights_path is not None:
        state_dict = torch.load(
            weights_path,
            weights_only=True,
            map_location="cpu",
        )["state_dict"]
        state_dict = {
            key.replace("unet.", ""): value for key, value in state_dict.items()
        }
        pipe.pipe.unet.load_state_dict(state_dict)
    return pipe
