from typing import Dict

from .v1 import StableDiffusionV1

__all__ = [
    "model_path",
    "StableDiffusionV2",
]


model_path: Dict[str, str] = {
    "2-base": "stabilityai/stable-diffusion-2-base",
    "2": "stabilityai/stable-diffusion-2",
    "2.1-base": "stabilityai/stable-diffusion-2-1-base",
    "2.1": "stabilityai/stable-diffusion-2-1",
}


class StableDiffusionV2(StableDiffusionV1):
    def __init__(
        self,
        version: str,
        scheduler: str,
        variant: str = "fp16",
        verbose: bool = False,
    ) -> None:
        super().__init__(version, scheduler, variant, verbose)
