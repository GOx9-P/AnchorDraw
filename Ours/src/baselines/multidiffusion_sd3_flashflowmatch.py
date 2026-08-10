from __future__ import annotations

import contextlib
from typing import Sequence

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm


def get_sd3_views(
    image_height: int,
    image_width: int,
    vae_scale_factor: int = 8,
    window_size: int = 128,
    stride: int = 128,
) -> list[tuple[int, int, int, int]]:
    """Return MultiDiffusion views in latent coordinates."""
    latent_height = image_height // vae_scale_factor
    latent_width = image_width // vae_scale_factor
    if latent_height <= window_size and latent_width <= window_size:
        return [(0, latent_height, 0, latent_width)]

    def starts(size: int) -> list[int]:
        if size <= window_size:
            return [0]
        values = list(range(0, size - window_size + 1, stride))
        last = size - window_size
        if values[-1] != last:
            values.append(last)
        return values

    views = []
    for h_start in starts(latent_height):
        h_end = min(h_start + window_size, latent_height)
        for w_start in starts(latent_width):
            w_end = min(w_start + window_size, latent_width)
            views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusionSD3FlashFlowMatch:
    """Region-based MultiDiffusion fusion with SD3 and Flash Flow Match.

    The region/window fusion follows the original MultiDiffusion
    `region_based.py`: masks and prompts include the background as the first
    region, foreground regions are denoised independently, and all region
    predictions are averaged by their masks.

    The SD3 model/scheduler path follows SemanticDraw's SD3 implementation:
    Stable Diffusion 3 Medium, `jasperai/flash-sd3`, and
    FlashFlowMatchEulerDiscreteScheduler.
    """

    def __init__(
        self,
        device: torch.device,
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        flash_sd3_repo_id: str = "jasperai/flash-sd3",
        dtype: torch.dtype = torch.float16,
        t_index_list: Sequence[int] = (0, 4, 12, 25, 37),
        schedule_steps: int = 50,
        safe_vae: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        view_window_size: int = 128,
        view_stride: int = 128,
        runtime_checks: bool = True,
        show_progress: bool = False,
    ) -> None:
        from diffusers import StableDiffusion3Pipeline
        from diffusers.models.transformers import SD3Transformer2DModel
        from diffusers.schedulers import FlashFlowMatchEulerDiscreteScheduler
        from peft import PeftModel

        self.device = device
        self.model_id = model_id
        self.flash_sd3_repo_id = flash_sd3_repo_id
        self.dtype = dtype
        self.t_index_list = list(t_index_list)
        self.schedule_steps = int(schedule_steps)
        self.safe_vae = bool(safe_vae)
        self.view_window_size = int(view_window_size)
        self.view_stride = int(view_stride)
        self.runtime_checks = bool(runtime_checks)
        self.show_progress = bool(show_progress)

        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        ).to(device)
        transformer = PeftModel.from_pretrained(transformer, flash_sd3_repo_id).to(device)

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype,
            text_encoder_3=None,
            tokenizer_3=None,
        ).to(device)
        self.pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )

        if enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if enable_vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))
        self.latent_scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.5305))
        self.latent_shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0609))

        self.prepare_flashflowmatch_schedule(self.t_index_list, self.schedule_steps)

    @staticmethod
    def _module_dtype(module: torch.nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def _assert_finite(self, name: str, tensor: torch.Tensor) -> None:
        if self.runtime_checks and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN/Inf; this is a wrapper/runtime error.")

    @torch.no_grad()
    def prepare_flashflowmatch_schedule(
        self,
        t_index_list: Sequence[int] | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        if t_index_list is None:
            t_index_list = self.t_index_list
        if num_inference_steps is None:
            num_inference_steps = self.schedule_steps

        try:
            self.scheduler.set_timesteps(int(num_inference_steps), device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(int(num_inference_steps))

        indices = torch.as_tensor(list(t_index_list), dtype=torch.long)
        max_index = len(self.scheduler.timesteps) - 1
        if indices.numel() == 0:
            raise ValueError("t_index_list must contain at least one index")
        if int(indices.min().item()) < 0 or int(indices.max().item()) > max_index:
            raise ValueError(f"t_index_list out of scheduler range 0..{max_index}: {list(t_index_list)}")

        self.timesteps = self.scheduler.timesteps[indices].to(device=self.device)
        self.sigmas = self.scheduler.sigmas[indices].to(device=self.device, dtype=torch.float32)
        self.sigmas_next = torch.cat([self.sigmas, self.sigmas.new_zeros(1)])[1:].to(device=self.device)

    def scheduler_step(self, noise_pred: torch.Tensor, index: int, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(torch.float32)
        noise_pred = noise_pred.to(torch.float32)
        prev_sample = latent - noise_pred * self.sigmas[index]
        return prev_sample.to(dtype=self.dtype)

    def scheduler_add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor | None,
        index: int,
    ) -> torch.Tensor:
        if index < 0 or index >= len(self.sigmas):
            return latent
        noise = torch.randn_like(latent) if noise is None else noise
        sigma = self.sigmas[index].to(device=latent.device, dtype=latent.dtype)
        return (1.0 - sigma) * latent + sigma * noise

    @torch.no_grad()
    def encode_imgs(self, images: torch.Tensor) -> torch.Tensor:
        images = 2 * images - 1
        original_dtype = self._module_dtype(self.vae)
        encode_dtype = torch.float32 if self.safe_vae else self.dtype

        if self.safe_vae and original_dtype != torch.float32:
            self.vae.to(dtype=torch.float32)
        try:
            posterior = self.vae.encode(images.to(device=self.device, dtype=encode_dtype)).latent_dist
            latents = posterior.sample() * self.latent_scaling_factor
            self._assert_finite("encoded VAE latents", latents)
        finally:
            if self.safe_vae and original_dtype != torch.float32:
                self.vae.to(dtype=original_dtype)
        return latents.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        original_dtype = self._module_dtype(self.vae)
        decode_dtype = torch.float32 if self.safe_vae else self.dtype

        if self.safe_vae and original_dtype != torch.float32:
            self.vae.to(dtype=torch.float32)
        try:
            latents = latents.to(device=self.device, dtype=decode_dtype)
            latents = (latents / self.latent_scaling_factor) + self.latent_shift_factor
            self._assert_finite("decoded VAE input latents", latents)
            images = self.vae.decode(latents, return_dict=False)[0]
            self._assert_finite("decoded VAE image tensor", images)
        finally:
            if self.safe_vae and original_dtype != torch.float32:
                self.vae.to(dtype=original_dtype)
        return (images.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def get_random_background(self, n_samples: int, height: int, width: int) -> torch.Tensor:
        in_channels = int(self.transformer.config.in_channels)
        if n_samples <= 0:
            return torch.empty(
                (0, in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
                dtype=self.dtype,
                device=self.device,
            )

        backgrounds = torch.rand(n_samples, 3, dtype=self.dtype, device=self.device)[:, :, None, None]
        backgrounds = backgrounds.repeat(1, 1, height, width)
        return torch.cat([self.encode_imgs(background.unsqueeze(0)) for background in backgrounds], dim=0)

    @torch.no_grad()
    def get_sd3_text_conditioning(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        guidance_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        do_classifier_free_guidance = guidance_scale > 1.0
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=self.device,
            clip_skip=getattr(self.pipe, "clip_skip", None),
            num_images_per_prompt=1,
        )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=self.dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        self._assert_finite("prompt embeddings", prompt_embeds)
        self._assert_finite("pooled prompt embeddings", pooled_prompt_embeds)
        return prompt_embeds, pooled_prompt_embeds, do_classifier_free_guidance

    @torch.no_grad()
    def generate(
        self,
        masks: torch.Tensor,
        prompts: list[str],
        negative_prompts: list[str] | str,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 0.0,
        bootstrapping: int = 2,
        t_index_list: Sequence[int] | None = None,
        schedule_steps: int | None = None,
    ) -> Image.Image:
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)
        if len(prompts) != int(masks.shape[0]):
            raise ValueError(f"prompts/masks mismatch: {len(prompts)} prompts vs {int(masks.shape[0])} masks")
        if len(negative_prompts) != len(prompts):
            raise ValueError(
                f"negative_prompts should match prompts: {len(negative_prompts)} vs {len(prompts)}"
            )
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"height/width must be divisible by {self.vae_scale_factor}: {(height, width)}")

        self.prepare_flashflowmatch_schedule(
            self.t_index_list if t_index_list is None else t_index_list,
            self.schedule_steps if schedule_steps is None else int(schedule_steps),
        )

        self.pipe._guidance_scale = guidance_scale
        self.pipe._clip_skip = getattr(self.pipe, "clip_skip", None)
        self.pipe._joint_attention_kwargs = None
        self.pipe._interrupt = False

        num_regions = len(prompts)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        in_channels = int(self.transformer.config.in_channels)

        masks = masks.to(device=self.device, dtype=self.dtype)
        if tuple(masks.shape[-2:]) != (latent_h, latent_w):
            masks = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")
        masks = masks.clamp(0, 1)

        bootstrapping = int(bootstrapping)
        bootstrapping_backgrounds = self.get_random_background(bootstrapping, height, width)
        self._assert_finite("bootstrapping backgrounds", bootstrapping_backgrounds)
        prompt_embeds, pooled_prompt_embeds, do_cfg = self.get_sd3_text_conditioning(
            prompts,
            list(negative_prompts),
            guidance_scale,
        )

        latent = torch.randn(
            (1, in_channels, latent_h, latent_w),
            dtype=self.dtype,
            device=self.device,
        )
        self._assert_finite("initial latent", latent)
        region_noise = latent.repeat(max(num_regions - 1, 1), 1, 1, 1)

        views = get_sd3_views(
            height,
            width,
            vae_scale_factor=self.vae_scale_factor,
            window_size=self.view_window_size,
            stride=self.view_stride,
        )
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
            else contextlib.nullcontext()
        )

        progress = tqdm(self.timesteps, leave=False) if self.show_progress else self.timesteps
        with autocast_ctx:
            for step_index, timestep in enumerate(progress):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(num_regions, 1, 1, 1)

                    if step_index < bootstrapping and num_regions > 1:
                        random_indices = torch.randint(
                            0,
                            bootstrapping,
                            (num_regions - 1,),
                            device=self.device,
                        )
                        background = bootstrapping_backgrounds[
                            random_indices,
                            :,
                            h_start:h_end,
                            w_start:w_end,
                        ]
                        background_noise = region_noise[: num_regions - 1, :, h_start:h_end, w_start:w_end]
                        background = self.scheduler_add_noise(background, background_noise, step_index)
                        self._assert_finite("bootstrapped background latent", background)
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + background * (1 - masks_view[1:])

                    latent_model_input = torch.cat([latent_view] * 2) if do_cfg else latent_view
                    timestep_input = timestep.expand(latent_model_input.shape[0])
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    self._assert_finite("transformer noise prediction", noise_pred)

                    if do_cfg:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    self._assert_finite("guided noise prediction", noise_pred)

                    latents_view_denoised = self.scheduler_step(noise_pred, step_index, latent_view)
                    self._assert_finite("denoised view latent", latents_view_denoised)
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdim=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdim=True)

                latent = torch.where(count > 0, value / count.clamp_min(1e-6), value)
                self._assert_finite("mixed latent", latent)
                if step_index < len(self.timesteps) - 1:
                    latent = self.scheduler_add_noise(latent, None, step_index + 1)
                    self._assert_finite("renoised latent", latent)

        images = self.decode_latents(latent)
        return T.ToPILImage()(images[0].detach().cpu())
