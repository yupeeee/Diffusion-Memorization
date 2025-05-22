from typing import Dict

import diffusers
import torch
import transformers

__all__ = [
    "_load_device",
    "_load_generator",
    "_load_pipe",
]


def _load_device(
    device: torch.device,
) -> torch.device:
    if isinstance(device, int) or isinstance(device, str):
        device = torch.device(device)
    return device


def _load_generator(
    seed: int = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Generator:
    if seed is None:
        return None
    else:
        return torch.Generator(device=device).manual_seed(seed)


def _is_valid_version(
    version: str,
    model_path: Dict[str, str],
) -> bool:
    if version not in model_path.keys():
        raise ValueError(
            f"Unsupported version: {version} ("
            + f"Supported versions: {', '.join(model_path.keys())})"
        )
    return True


def _load_scheduler(
    version: str,
    strategy: str,
    model_path: Dict[str, str],
) -> diffusers.schedulers.scheduling_utils.SchedulerMixin:
    try:
        scheduler = getattr(diffusers, f"{strategy}Scheduler").from_pretrained(
            pretrained_model_name_or_path=model_path[version],
            subfolder="scheduler",
        )
        return scheduler
    except AttributeError as e:
        raise ValueError(f"Unknown scheduler: {strategy} ({e})")


def _load_image_encoder(
    version: str,
    model_path: Dict[str, str],
) -> transformers.CLIPVisionModelWithProjection:
    image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        # subfolder="image_encoder",
    )
    return image_encoder


def _load_pipe(
    version: str,
    scheduler: str,
    model_path: Dict[str, str],
    variant: str = "fp16",
    verbose: bool = False,
) -> diffusers.StableDiffusionPipeline:
    assert _is_valid_version(version, model_path)
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path[version],
        scheduler=_load_scheduler(version, scheduler, model_path),
        image_encoder=_load_image_encoder(version, model_path),
        variant=variant,
    )
    pipe.safety_checker = None
    if not verbose:
        pipe.set_progress_bar_config(disable=True)
    return pipe
