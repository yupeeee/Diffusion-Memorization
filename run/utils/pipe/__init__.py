from typing import Union

from . import v1, v2
from ._unet_layers import unet_layers

__all__ = [
    "unet_layers",
    "StableDiffusion",
]


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
