import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from diffusers import UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from PIL.Image import Image
from torch import Tensor

from .load import _load_device, _load_generator, _load_pipe

__all__ = [
    "model_path",
    "StableDiffusionV1",
]

model_path: Dict[str, str] = {
    "1.1": "CompVis/stable-diffusion-v1-1",
    "1.2": "CompVis/stable-diffusion-v1-2",
    "1.3": "CompVis/stable-diffusion-v1-3",
    "1.4": "CompVis/stable-diffusion-v1-4",
    "1.5": "sd-legacy/stable-diffusion-v1-5",
}

IMG_SIZE = 32 * 8


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionV1:
    def __init__(
        self,
        version: str,
        scheduler: str,
        variant: str = "fp16",
        verbose: bool = False,
    ) -> None:
        self.pipe = _load_pipe(version, scheduler, model_path, variant, verbose)
        self.device = _load_device(self.pipe.device)
        self.generator = _load_generator(device=self.device)

        # Change U-Net
        unet_config = dict(self.pipe.unet.config)
        unet_config.update(
            {
                "sample_size": (IMG_SIZE, IMG_SIZE),
                "in_channels": 3,
                "out_channels": 3,
                # "block_out_channels": [
                #     64,
                #     128,
                #     256,
                #     256,
                # ],
            }
        )
        self.pipe.unet = UNet2DConditionModel(**unet_config).to(self.device)

        # Change VAE
        self.pipe.vae = None

        self.config = {
            "height": IMG_SIZE,
            "width": IMG_SIZE,
            "lora_scale": None,
            "timesteps": None,
            "num_inference_steps": 50,
            "extra_step_kwargs": None,
        }

    def to(
        self,
        device: torch.device,
    ) -> "StableDiffusionV1":
        self.device = _load_device(device)
        self.pipe.to(self.device)
        self.generator = _load_generator(device=self.device)
        return self

    def seed(
        self,
        seed: int = None,
    ) -> "StableDiffusionV1":
        self.generator = _load_generator(seed=seed, device=self.device)
        return self

    def alphas_cumprod(
        self,
        num_inference_steps: int = 50,
    ) -> Tensor:
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        alphas_cumprod = []
        for step in range(0, num_inference_steps):
            t = self.pipe.scheduler.timesteps[step]
            alphas_cumprod.append(self.pipe.scheduler.alphas_cumprod[t.item()])
        return torch.stack(alphas_cumprod, dim=0)  # (num_inference_steps, )

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        lora_scale: float = None,
        clip_skip: int = None,
    ) -> Tuple[Tensor, Tensor]:
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )  # ([B * num_images_per_prompt, 77, 768], [B * num_images_per_prompt, 77, 768])
        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image, List[Image]],
        num_images_per_prompt: int = 1,
        output_hidden_states: bool = None,
    ) -> Tuple[Tensor, Tensor]:
        image_embeds, uncond_image_embeds = self.pipe.encode_image(
            image=image,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            output_hidden_states=output_hidden_states,
        )  # ([B * num_images_per_prompt, 1, 768], [B * num_images_per_prompt, 1, 768])
        return image_embeds, uncond_image_embeds

    @torch.no_grad()
    def decode(
        self,
        latents: Tensor,
    ) -> List[Image]:
        pil_images = []
        transform = T.ToPILImage()
        for latent in latents:
            pil_images.append(transform(latent))
        return pil_images

    def classifier_free_guidance(
        self,
        noise_pred: Tensor,
        guidance_scale: float = None,
        guidance_rescale: float = None,
    ) -> Tensor:
        assert self.pipe.do_classifier_free_guidance

        if guidance_scale is None:
            guidance_scale = self.pipe.guidance_scale
        if guidance_rescale is None:
            guidance_rescale = self.pipe.guidance_rescale

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if guidance_rescale > 0.0:
            # Based on 3.4. in https://huggingface.co/papers/2305.08891
            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_text,
                guidance_rescale=guidance_rescale,
            )
        return noise_pred

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        ts: Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> Tensor:
        if num_inference_steps is not None:
            self.pipe.scheduler.set_timesteps(num_inference_steps)

        return self.pipe.scheduler.add_noise(
            original_samples=original_samples.to(self.device),
            noise=noise.to(self.device),
            timesteps=ts.to(self.device),
        )

    def predict_noise(
        self,
        noisy_samples: Tensor,
        ts: Tensor,
        prompt_embeds: Tensor,
        num_inference_steps: Optional[int] = None,
        timestep_cond: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        # expand the latents if we are doing classifier free guidance
        if self.pipe.do_classifier_free_guidance:
            noisy_samples = torch.cat([noisy_samples] * 2)
            ts = ts.repeat(2)
            prompt_embeds = torch.cat([prompt_embeds] * 2)

        noisy_samples = self.pipe.scheduler.scale_model_input(noisy_samples, ts)

        if num_inference_steps is not None:
            self.pipe.scheduler.set_timesteps(num_inference_steps)

        noise_pred = self.pipe.unet(
            noisy_samples,
            ts,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        return noise_pred

    def init(
        self,
        height: Optional[int] = IMG_SIZE,
        width: Optional[int] = IMG_SIZE,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
    ) -> None:
        self.config = {}

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.pipe.unet.config.sample_size
                if self.pipe._is_unet_config_sample_size_int
                else self.pipe.unet.config.sample_size[0]
            )
            width = (
                self.pipe.unet.config.sample_size
                if self.pipe._is_unet_config_sample_size_int
                else self.pipe.unet.config.sample_size[1]
            )
            height, width = (
                height * self.pipe.vae_scale_factor,
                width * self.pipe.vae_scale_factor,
            )
        # to deal with lora scaling and other possible forward hooks
        self.config["height"] = height
        self.config["width"] = width

        self.pipe._guidance_scale = guidance_scale
        self.pipe._guidance_rescale = guidance_rescale
        self.pipe._clip_skip = clip_skip
        self.pipe._cross_attention_kwargs = cross_attention_kwargs
        self.pipe._interrupt = False

        # 3. Encode input prompt
        lora_scale = (
            self.pipe.cross_attention_kwargs.get("scale", None)
            if self.pipe.cross_attention_kwargs is not None
            else None
        )
        self.config["lora_scale"] = lora_scale

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, self.device, timesteps, sigmas
        )
        self.config["timesteps"] = timesteps
        self.config["num_inference_steps"] = num_inference_steps

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(self.generator, eta)
        self.config["extra_step_kwargs"] = extra_step_kwargs

    @torch.no_grad()
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
    ) -> Tuple[List[Image], List[Tensor], List[Tensor]]:
        prompt_embeds: Optional[Tensor] = kwargs.pop("prompt_embeds", None)
        negative_prompt_embeds: Optional[Tensor] = kwargs.pop(
            "negative_prompt_embeds", None
        )
        ip_adapter_image: Optional[PipelineImageInput] = kwargs.pop(
            "ip_adapter_image", None
        )
        ip_adapter_image_embeds: Optional[List[Tensor]] = kwargs.pop(
            "ip_adapter_image_embeds", None
        )

        if seed is not None:
            self.seed(seed)

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(
            prompt,
            self.config["height"],
            self.config["width"],
            None,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            None,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            self.pipe.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=self.config["lora_scale"],
            clip_skip=self.pipe.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                batch_size * num_images_per_prompt,
                self.pipe.do_classifier_free_guidance,
            )

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            self.config["height"],
            self.config["width"],
            prompt_embeds.dtype,
            self.device,
            self.generator,
            latents,
        )

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = Tensor(self.pipe.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.pipe.unet.config.time_cond_proj_dim,
            ).to(device=self.device, dtype=latents.dtype)

        # 7. Denoising loop
        x_ts = [latents.cpu()]
        eps_ts = []

        if start_step is None:
            start_step = 0
        if exit_step is None:
            exit_step = self.config["num_inference_steps"]
        timesteps = self.config["timesteps"][start_step:exit_step]

        num_warmup_steps = (
            len(timesteps)
            - self.config["num_inference_steps"] * self.pipe.scheduler.order
        )
        self.pipe._num_timesteps = len(timesteps)
        with self.pipe.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.pipe.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                eps_ts.append(noise_pred.cpu())

                # perform guidance
                if self.pipe.do_classifier_free_guidance:
                    noise_pred = self.classifier_free_guidance(noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **self.config["extra_step_kwargs"],
                    return_dict=False,
                )[0]

                x_ts.append(latents.cpu())

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.pipe.scheduler.order == 0
                ):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        images = self.decode(latents)

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        return images, x_ts, eps_ts
