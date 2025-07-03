from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .misc import get_submodule
from .pipe import SDPipeline, unet_layers

__all__ = [
    "UNetExtractor",
]

_unet_layers = unet_layers()


class UNetExtractor:
    def __init__(
        self,
        pipe: SDPipeline,
    ) -> None:
        self.pipe = pipe
        self.features: Dict[str, Tensor]
        self.layers: Dict[nn.Module, str] = dict(
            (get_submodule(self.pipe.pipe.unet, layer), layer) for layer in _unet_layers
        )

    def _hook_fn(
        self,
        module: nn.Module,
        input: Tensor,
        output: Tensor,
    ) -> None:
        output = (
            output[0] if isinstance(output, tuple) else output
        )  # (2 * batch_size, *feature_dim)
        self.features[self.layers[module]].append(output.cpu())

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[Tensor] = None,
        seed: Optional[int] = None,
        start_step: Optional[int] = None,
        exit_step: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        # Reset features
        self.features = dict((layer, []) for layer in _unet_layers)

        # Register hooks
        hooks = []
        for layer in _unet_layers:
            hook = get_submodule(self.pipe.pipe.unet, layer).register_forward_hook(
                self._hook_fn
            )
            hooks.append(hook)

        # Run pipeline
        _ = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            latents=latents,
            seed=seed,
            start_step=start_step,
            exit_step=exit_step,
            **kwargs,
        )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        for layer, features in self.features.items():
            self.features[layer] = torch.stack(features, dim=0)

        return self.features
