from __future__ import annotations

import contextlib
from typing import Sequence

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm

from .multidiffusion_lcm import get_views


class MultiDiffusionSDXLEuler:
    """Region-based MultiDiffusion fusion with SDXL-Lightning Euler sampling.

    The region/window fusion follows the original MultiDiffusion
    `region_based.py`: the first mask/prompt pair is the background region,
    foreground regions are denoised independently inside each sliding window,
    and all region predictions are averaged by their masks.

    The denoising core intentionally follows the SemanticDraw SDXL sampler
    family: Stable Diffusion XL base, SDXL-Lightning 4-step UNet, and
    EulerDiscreteScheduler with trailing spacing.
    """

    def __init__(
        self,
        device: torch.device,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lightning_repo_id: str = "ByteDance/SDXL-Lightning",
        lightning_weight_name: str = "sdxl_lightning_4step_unet.safetensors",
        dtype: torch.dtype = torch.float16,
        variant: str | None = "fp16",
        timestep_spacing: str = "trailing",
        t_index_list: Sequence[int] = (0, 4, 12, 25, 37),
        schedule_steps: int = 50,
        safe_vae: bool = True,
    ) -> None:
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.device = device
        self.model_id = model_id
        self.lightning_repo_id = lightning_repo_id
        self.lightning_weight_name = lightning_weight_name
        self.dtype = dtype
        self.variant = variant
        self.timestep_spacing = timestep_spacing
        self.t_index_list = list(t_index_list)
        self.schedule_steps = int(schedule_steps)
        self.safe_vae = bool(safe_vae)

        unet = UNet2DConditionModel.from_config(model_id, subfolder="unet").to(device, dtype)
        weight_path = hf_hub_download(
            repo_id=lightning_repo_id,
            filename=lightning_weight_name,
        )
        state_dict = load_file(weight_path, device="cpu")
        unet.load_state_dict(state_dict)
        unet.to(device, dtype)

        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                unet=unet,
                torch_dtype=dtype,
                variant=variant,
            ).to(device)
        except Exception:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                unet=unet,
                torch_dtype=dtype,
            ).to(device)

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing=timestep_spacing,
        )

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
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

        self.prepare_lightning_schedule(self.t_index_list, self.schedule_steps)

    @staticmethod
    def _module_dtype(module: torch.nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    @torch.no_grad()
    def prepare_lightning_schedule(
        self,
        t_index_list: Sequence[int] | None = None,
        num_inference_steps: int | None = None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
    ) -> None:
        if t_index_list is None:
            t_index_list = self.t_index_list
        if num_inference_steps is None:
            num_inference_steps = self.schedule_steps

        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)

        indices = torch.as_tensor(list(t_index_list), dtype=torch.long)
        max_index = len(self.scheduler.timesteps) - 1
        if indices.numel() == 0:
            raise ValueError("t_index_list must contain at least one index")
        if int(indices.min().item()) < 0 or int(indices.max().item()) > max_index:
            raise ValueError(f"t_index_list out of scheduler range 0..{max_index}: {list(t_index_list)}")

        self.timesteps = self.scheduler.timesteps[indices].to(device=self.device)
        self.sigmas = self.scheduler.sigmas[indices].to(device=self.device, dtype=torch.float32)
        self.sigmas_next = torch.cat([self.sigmas, self.sigmas.new_zeros(1)])[1:]

        sigma_mask = torch.logical_and(s_tmin <= self.sigmas, self.sigmas <= s_tmax)
        if num_inference_steps <= 1:
            gamma_value = 0.0
        else:
            gamma_value = min(float(s_churn) / (int(num_inference_steps) - 1), 2**0.5 - 1)
        self.gammas = gamma_value * sigma_mask.to(dtype=torch.float32)
        self.sigma_hats = self.sigmas * (self.gammas + 1.0)
        self.dt = self.sigmas_next - self.sigma_hats

    def scheduler_scale_model_input(self, latent: torch.Tensor, index: int) -> torch.Tensor:
        return latent / ((self.sigmas[index] ** 2 + 1.0) ** 0.5)

    def scheduler_step(self, noise_pred: torch.Tensor, index: int, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(torch.float32)
        noise_pred = noise_pred.to(torch.float32)
        prev_sample = latent + noise_pred * self.dt[index]
        return prev_sample.to(self.dtype)

    def scheduler_add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor | None,
        index: int,
        s_noise: float = 1.0,
        initial: bool = False,
    ) -> torch.Tensor:
        if index < 0 or index >= len(self.sigmas):
            return latent

        noise = torch.randn_like(latent) if noise is None else noise
        if initial:
            return latent + self.sigmas[index].to(latent.dtype) * noise

        noise_level = (self.sigma_hats[index] ** 2 - self.sigmas[index] ** 2).clamp_min(0).sqrt()
        if self.gammas[index] > 0 and noise_level > 0 and s_noise > 0:
            return latent + noise * float(s_noise) * noise_level.to(latent.dtype)
        return latent

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
            images = self.vae.decode(latents, return_dict=False)[0]
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
        return torch.cat([self.encode_imgs(background.unsqueeze(0)) for background in backgrounds], dim=0)

    @torch.no_grad()
    def generate(
        self,
        masks: torch.Tensor,
        prompts: list[str],
        negative_prompts: list[str] | str,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 1.0,
        bootstrapping: int = 1,
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

        self.prepare_lightning_schedule(
            self.t_index_list if t_index_list is None else t_index_list,
            self.schedule_steps if schedule_steps is None else int(schedule_steps),
        )

        num_regions = len(prompts)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        masks = masks.to(device=self.device, dtype=self.dtype)
        if tuple(masks.shape[-2:]) != (latent_h, latent_w):
            masks = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")
        masks = masks.clamp(0, 1)

        bootstrapping_backgrounds = self.get_random_background(int(bootstrapping), height, width)
        prompt_embeds, add_text_embeds, add_time_ids = self.get_sdxl_text_conditioning(
            prompts,
            list(negative_prompts),
            height,
            width,
        )

        latent = torch.randn(
            (1, self.unet.config.in_channels, latent_h, latent_w),
            dtype=self.dtype,
            device=self.device,
        )
        latent = latent * float(self.scheduler.init_noise_sigma)
        region_noise = latent.clone().repeat(max(num_regions - 1, 1), 1, 1, 1)

        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            for step_index, timestep in enumerate(tqdm(self.timesteps, leave=False)):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(num_regions, 1, 1, 1)

                    if step_index < bootstrapping and num_regions > 1:
                        random_indices = torch.randint(
                            0,
                            int(bootstrapping),
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
                        background = self.scheduler_add_noise(
                            background,
                            background_noise,
                            step_index,
                            initial=True,
                        )
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + background * (1 - masks_view[1:])

                    latent_model_input = torch.cat([latent_view] * 2)
                    latent_model_input = self.scheduler_scale_model_input(latent_model_input, step_index)

                    add_time_ids_input = add_time_ids.clone()
                    add_time_ids_input[:, 2] = h_start * self.vae_scale_factor
                    add_time_ids_input[:, 3] = w_start * self.vae_scale_factor
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids_input,
                    }
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents_view_denoised = self.scheduler_step(noise_pred, step_index, latent_view)
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdim=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdim=True)

                latent = torch.where(count > 0, value / count.clamp_min(1e-6), latent)
                if step_index < len(self.timesteps) - 1:
                    latent = self.scheduler_add_noise(latent, None, step_index + 1)

        images = self.decode_latents(latent)
        return T.ToPILImage()(images[0].detach().cpu())
