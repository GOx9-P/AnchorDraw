from __future__ import annotations

import contextlib

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm

from .multidiffusion_sdxl_euler import get_sdxl_views


class MultiDiffusionSDXLDDIM:
    """Region-based MultiDiffusion fusion with SDXL base and DDIM sampling.

    The region fusion and random-background bootstrapping follow
    `Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py`:
    masks and prompts include the background as the first region, foreground
    regions are bootstrapped with random-color background latents during early
    denoising steps, and per-view region predictions are fused by masks.

    The SDXL-specific parts (dual text conditioning, pooled text embeddings,
    and time ids) are handled through Diffusers' StableDiffusionXLPipeline
    APIs so the UNet receives the conditioning tensors it expects.
    """

    def __init__(
        self,
        device: torch.device,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        dtype: torch.dtype = torch.float16,
        variant: str | None = "fp16",
        safe_vae: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        view_window_size: int = 64,
        view_stride: int = 8,
        runtime_checks: bool = True,
        show_progress: bool = False,
    ) -> None:
        from diffusers import DDIMScheduler, StableDiffusionXLPipeline

        self.device = device
        self.model_id = model_id
        self.dtype = dtype
        self.variant = variant
        self.safe_vae = bool(safe_vae)
        self.view_window_size = int(view_window_size)
        self.view_stride = int(view_stride)
        self.runtime_checks = bool(runtime_checks)
        self.show_progress = bool(show_progress)

        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant=variant,
            ).to(device)
        except Exception:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        if enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if enable_vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))
        self.latent_scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.13025))

    @staticmethod
    def _module_dtype(module: torch.nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def _assert_finite(self, name: str, tensor: torch.Tensor) -> None:
        if self.runtime_checks and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN/Inf; this is a wrapper/runtime error.")

    def _get_add_time_ids(
        self,
        original_size: tuple[int, int],
        crops_coords_top_left: tuple[int, int],
        target_size: tuple[int, int],
        dtype: torch.dtype,
        text_encoder_projection_dim: int,
    ) -> torch.Tensor:
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                "SDXL added time embedding dimension mismatch: "
                f"expected {expected_add_embed_dim}, got {passed_add_embed_dim}"
            )
        return torch.tensor([add_time_ids], dtype=dtype)

    @torch.no_grad()
    def get_sdxl_text_conditioning(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
        )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)

        text_encoder_projection_dim = (
            int(pooled_prompt_embeds.shape[-1])
            if self.text_encoder_2 is None
            else int(self.text_encoder_2.config.projection_dim)
        )
        add_time_ids = self._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(self.device)
        add_time_ids = add_time_ids.repeat(len(prompts), 1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        return prompt_embeds, add_text_embeds, add_time_ids

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
        decode_dtype = torch.float32 if getattr(self.vae.config, "force_upcast", False) else self.dtype

        if decode_dtype == torch.float32 and original_dtype != torch.float32:
            self.vae.to(dtype=torch.float32)
        try:
            latents = latents.to(device=self.device, dtype=decode_dtype)
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(
                    self.vae.config.latents_mean,
                    device=self.device,
                    dtype=latents.dtype,
                ).view(1, 4, 1, 1)
                latents_std = torch.tensor(
                    self.vae.config.latents_std,
                    device=self.device,
                    dtype=latents.dtype,
                ).view(1, 4, 1, 1)
                latents = latents * latents_std / self.latent_scaling_factor + latents_mean
            else:
                latents = latents / self.latent_scaling_factor
            self._assert_finite("decoded VAE input latents", latents)
            images = self.vae.decode(latents, return_dict=False)[0]
            self._assert_finite("decoded VAE image tensor", images)
        finally:
            if decode_dtype == torch.float32 and original_dtype != torch.float32:
                self.vae.to(dtype=original_dtype)
        return (images / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def get_random_background(self, n_samples: int, height: int, width: int) -> torch.Tensor:
        if n_samples <= 0:
            return torch.empty(
                (0, self.unet.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
                dtype=self.dtype,
                device=self.device,
            )

        backgrounds = torch.rand(n_samples, 3, dtype=self.dtype, device=self.device)[:, :, None, None]
        backgrounds = backgrounds.repeat(1, 1, height, width)

        original_dtype = self._module_dtype(self.vae)
        encode_dtype = torch.float32 if self.safe_vae else self.dtype
        if self.safe_vae and original_dtype != torch.float32:
            self.vae.to(dtype=torch.float32)
        try:
            latents = []
            for background in backgrounds:
                image = 2 * background.unsqueeze(0) - 1
                posterior = self.vae.encode(image.to(device=self.device, dtype=encode_dtype)).latent_dist
                latent = posterior.sample() * self.latent_scaling_factor
                self._assert_finite("encoded random background latent", latent)
                latents.append(latent.to(device=self.device, dtype=self.dtype))
        finally:
            if self.safe_vae and original_dtype != torch.float32:
                self.vae.to(dtype=original_dtype)
        return torch.cat(latents, dim=0)

    def scheduler_add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = torch.full(
            (clean_latents.shape[0],),
            int(timestep.item()),
            dtype=torch.long,
            device=self.device,
        )
        return self.scheduler.add_noise(clean_latents, noise, timesteps)

    def scheduler_step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latent_view: torch.Tensor,
    ) -> torch.Tensor:
        try:
            result = self.scheduler.step(
                noise_pred,
                timestep,
                latent_view,
                eta=0.0,
                return_dict=True,
            )
        except TypeError:
            result = self.scheduler.step(noise_pred, timestep, latent_view, return_dict=True)
        return result.prev_sample

    @torch.no_grad()
    def generate(
        self,
        masks: torch.Tensor,
        prompts: list[str],
        negative_prompts: list[str] | str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        bootstrapping: int = 20,
        view_window_size: int | None = None,
        view_stride: int | None = None,
        show_progress: bool | None = None,
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

        num_regions = len(prompts)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        masks = masks.to(device=self.device, dtype=self.dtype)
        if tuple(masks.shape[-2:]) != (latent_h, latent_w):
            masks = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")
        masks = masks.clamp(0, 1)

        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        views = get_sdxl_views(
            height,
            width,
            vae_scale_factor=self.vae_scale_factor,
            window_size=self.view_window_size if view_window_size is None else int(view_window_size),
            stride=self.view_stride if view_stride is None else int(view_stride),
        )

        bootstrapping = int(bootstrapping)
        window_latent_h = max(h_end - h_start for h_start, h_end, _, _ in views)
        window_latent_w = max(w_end - w_start for _, _, w_start, w_end in views)
        bootstrapping_backgrounds = self.get_random_background(
            bootstrapping,
            window_latent_h * self.vae_scale_factor,
            window_latent_w * self.vae_scale_factor,
        )
        self._assert_finite("bootstrapping backgrounds", bootstrapping_backgrounds)
        prompt_embeds, add_text_embeds, add_time_ids = self.get_sdxl_text_conditioning(
            prompts,
            list(negative_prompts),
            height,
            width,
        )
        self._assert_finite("prompt embeddings", prompt_embeds)

        latent = torch.randn(
            (1, self.unet.config.in_channels, latent_h, latent_w),
            dtype=self.dtype,
            device=self.device,
        )
        latent = latent * float(self.scheduler.init_noise_sigma)
        region_noise = latent.clone().repeat(max(num_regions - 1, 1), 1, 1, 1)

        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        if show_progress is None:
            show_progress = self.show_progress
        timestep_iterable = tqdm(self.scheduler.timesteps, leave=False) if show_progress else self.scheduler.timesteps

        autocast_ctx = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            for step_index, timestep in enumerate(timestep_iterable):
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
                        view_h = h_end - h_start
                        view_w = w_end - w_start
                        background = bootstrapping_backgrounds[random_indices, :, :view_h, :view_w]
                        background_noise = region_noise[: num_regions - 1, :, h_start:h_end, w_start:w_end]
                        background = self.scheduler_add_noise(background, background_noise, timestep)
                        self._assert_finite("bootstrapped background latent", background)
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + background * (1 - masks_view[1:])

                    latent_model_input = torch.cat([latent_view] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    self._assert_finite("UNet noise prediction", noise_pred)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    self._assert_finite("guided noise prediction", noise_pred)

                    latents_view_denoised = self.scheduler_step(noise_pred, timestep, latent_view)
                    self._assert_finite("denoised view latent", latents_view_denoised)
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdim=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdim=True)

                latent = torch.where(count > 0, value / count.clamp_min(1e-6), value)
                self._assert_finite("mixed latent", latent)

        images = self.decode_latents(latent)
        return T.ToPILImage()(images[0].detach().cpu())
