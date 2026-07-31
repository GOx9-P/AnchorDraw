from __future__ import annotations

import contextlib

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm

from .multidiffusion_lcm import get_views


class MultiDiffusionDDIM:
    """Region-based MultiDiffusion with the original SD1.5 DDIM sampler.

    This class is intentionally close to
    `Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py`.
    It keeps the original region fusion, background-mask convention, random
    color bootstrapping, DDIM scheduler, and classifier-free guidance loop.

    Practical runtime knobs such as dtype and attention/VAE slicing are exposed
    for Colab/Kaggle compatibility, but they do not change the core
    MultiDiffusion denoising/mask-fusion logic.
    """

    def __init__(
        self,
        device: torch.device,
        sd_version: str = "1.5",
        model_id: str | None = None,
        dtype: torch.dtype | None = torch.float16,
        use_autocast: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        show_progress: bool = False,
    ) -> None:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.sd_version = sd_version
        self.dtype = dtype
        self.use_autocast = bool(use_autocast)
        self.show_progress = bool(show_progress)

        if model_id is not None:
            self.model_id = model_id
        elif sd_version == "2.1":
            self.model_id = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == "2.0":
            self.model_id = "stabilityai/stable-diffusion-2-base"
        elif sd_version == "1.5":
            self.model_id = "runwayml/stable-diffusion-v1-5"
        else:
            self.model_id = sd_version

        load_kwargs = {}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            **load_kwargs,
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            **load_kwargs,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            **load_kwargs,
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        if enable_attention_slicing and hasattr(self.unet, "set_attention_slice"):
            with contextlib.suppress(Exception):
                self.unet.set_attention_slice("auto")
        if enable_vae_slicing and hasattr(self.vae, "enable_slicing"):
            with contextlib.suppress(Exception):
                self.vae.enable_slicing()

        self.vae_scale_factor = 8
        self.latent_scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.in_channels = int(getattr(self.unet.config, "in_channels", 4))

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def _latent_dtype(self) -> torch.dtype:
        return self.dtype if self.dtype is not None else self.model_dtype

    @torch.no_grad()
    def get_random_background(self, n_samples: int, height: int = 512, width: int = 512) -> torch.Tensor:
        """Original MultiDiffusion random constant-color background bootstrap."""
        if n_samples <= 0:
            return torch.empty(
                (0, self.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
                dtype=self._latent_dtype(),
                device=self.device,
            )

        backgrounds = torch.rand(
            n_samples,
            3,
            dtype=self._latent_dtype(),
            device=self.device,
        )[:, :, None, None]
        backgrounds = backgrounds.repeat(1, 1, height, width)
        return torch.cat([self.encode_imgs(background.unsqueeze(0)) for background in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompts: list[str], negative_prompts: list[str]) -> torch.Tensor:
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            negative_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def encode_imgs(self, images: torch.Tensor) -> torch.Tensor:
        images = 2 * images - 1
        images = images.to(device=self.device, dtype=self._latent_dtype())
        posterior = self.vae.encode(images).latent_dist
        return posterior.sample() * self.latent_scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latent_scaling_factor
        images = self.vae.decode(latents.to(device=self.device, dtype=self._latent_dtype())).sample
        return (images / 2 + 0.5).clamp(0, 1)

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
        negative_prompts: list[str],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        bootstrapping: int = 20,
        show_progress: bool | None = None,
    ) -> Image.Image:
        if len(prompts) != int(masks.shape[0]):
            raise ValueError(f"prompts/masks mismatch: {len(prompts)} prompts vs {int(masks.shape[0])} masks")
        if len(negative_prompts) != len(prompts):
            raise ValueError(
                f"negative_prompts should match prompts: {len(negative_prompts)} vs {len(prompts)}"
            )

        num_regions = len(prompts)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_dtype = self._latent_dtype()

        masks = masks.to(device=self.device, dtype=latent_dtype)
        if tuple(masks.shape[-2:]) != (latent_h, latent_w):
            masks = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")
        masks = masks.clamp(0, 1)

        bootstrapping = int(bootstrapping)
        bootstrapping_backgrounds = self.get_random_background(bootstrapping, height, width)
        text_embeds = self.get_text_embeds(prompts, negative_prompts).to(dtype=latent_dtype)

        latent = torch.randn(
            (1, self.in_channels, latent_h, latent_w),
            dtype=latent_dtype,
            device=self.device,
        )
        region_noise = latent.clone().repeat(max(num_regions - 1, 1), 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        if show_progress is None:
            show_progress = self.show_progress
        timestep_iterable = tqdm(self.scheduler.timesteps, leave=False) if show_progress else self.scheduler.timesteps

        autocast_dtype = latent_dtype if latent_dtype in (torch.float16, torch.bfloat16) else torch.float16
        autocast_ctx = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if self.device.type == "cuda" and self.use_autocast
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
                        background = bootstrapping_backgrounds[random_indices]
                        background_noise = region_noise[: num_regions - 1, :, h_start:h_end, w_start:w_end]
                        background = self.scheduler_add_noise(background, background_noise, timestep)
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + background * (1 - masks_view[1:])

                    latent_model_input = torch.cat([latent_view] * 2)
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeds,
                    )["sample"]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents_view_denoised = self.scheduler_step(noise_pred, timestep, latent_view)
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdim=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdim=True)

                latent = torch.where(count > 0, value / count, value)

        images = self.decode_latents(latent)
        return T.ToPILImage()(images[0].detach().cpu())
