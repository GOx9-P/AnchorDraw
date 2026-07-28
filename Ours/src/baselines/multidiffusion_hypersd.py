from __future__ import annotations

import contextlib

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm

from .multidiffusion_lcm import get_views


class MultiDiffusionHyperSD:
    """Region-based MultiDiffusion fusion with SD1.5 Hyper-SD sampler.

    The region/window fusion follows the original MultiDiffusion
    `region_based.py`: each masked region is denoised independently per view,
    then all denoised views are averaged by their masks. The intentional change
    for this experiment is replacing the original DDIM 50-step baseline with
    Hyper-SD SD1.5 4-step LoRA plus DDIMScheduler trailing.
    """

    def __init__(
        self,
        model_id: str,
        hyper_sd_repo_id: str,
        hyper_sd_weight_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 4,
        lora_scale: float = 1.0,
    ) -> None:
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        from huggingface_hub import hf_hub_download

        self.model_id = model_id
        self.hyper_sd_repo_id = hyper_sd_repo_id
        self.hyper_sd_weight_name = hyper_sd_weight_name
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = int(num_inference_steps)
        self.lora_scale = float(lora_scale)

        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                variant="fp16",
                torch_dtype=dtype,
            ).to(device)
        except Exception:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)

        try:
            self.pipe.scheduler = DDIMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
                timestep_spacing="trailing",
            )
        except Exception:
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )

        hyper_sd_lora_path = hf_hub_download(
            repo_id=hyper_sd_repo_id,
            filename=hyper_sd_weight_name,
        )
        self.pipe.load_lora_weights(hyper_sd_lora_path, adapter_name="hyper_sd")
        if hasattr(self.pipe, "set_adapters"):
            try:
                self.pipe.set_adapters(["hyper_sd"], adapter_weights=[self.lora_scale])
            except TypeError:
                self.pipe.set_adapters(["hyper_sd"])
        if hasattr(self.pipe, "fuse_lora"):
            with contextlib.suppress(Exception):
                self.pipe.fuse_lora(lora_scale=self.lora_scale, safe_fusing=False)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))
        self.latent_scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))

        self.prepare_schedule(self.num_inference_steps)

    @torch.no_grad()
    def prepare_schedule(self, num_inference_steps: int) -> None:
        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = self.scheduler.timesteps.to(device=self.device)

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
        posterior = self.vae.encode(images.to(dtype=self.dtype, device=self.device)).latent_dist
        return posterior.sample() * self.latent_scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latent_scaling_factor
        images = self.vae.decode(latents.to(dtype=self.dtype)).sample
        return (images / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def get_random_background(self, n_samples: int, height: int, width: int) -> torch.Tensor:
        if n_samples <= 0:
            return torch.empty(
                (0, self.unet.config.in_channels, height // 8, width // 8),
                dtype=self.dtype,
                device=self.device,
            )

        backgrounds = torch.rand(n_samples, 3, dtype=self.dtype, device=self.device)[:, :, None, None]
        backgrounds = backgrounds.repeat(1, 1, height, width)
        return torch.cat([self.encode_imgs(background.unsqueeze(0)) for background in backgrounds])

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
        guidance_scale: float = 1.0,
        bootstrapping: int = 1,
        num_inference_steps: int | None = None,
    ) -> Image.Image:
        if len(prompts) != int(masks.shape[0]):
            raise ValueError(f"prompts/masks mismatch: {len(prompts)} prompts vs {int(masks.shape[0])} masks")
        if len(negative_prompts) != len(prompts):
            raise ValueError(
                f"negative_prompts should match prompts: {len(negative_prompts)} vs {len(prompts)}"
            )

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        self.prepare_schedule(int(num_inference_steps))

        num_regions = len(prompts)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        masks = masks.to(device=self.device, dtype=self.dtype)
        if tuple(masks.shape[-2:]) != (latent_h, latent_w):
            masks = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")
        masks = masks.clamp(0, 1)

        bootstrapping_backgrounds = self.get_random_background(int(bootstrapping), height, width)
        text_embeds = self.get_text_embeds(prompts, negative_prompts).to(dtype=self.dtype)

        latent = torch.randn(
            (1, self.unet.config.in_channels, latent_h, latent_w),
            dtype=self.dtype,
            device=self.device,
        )
        region_noise = latent.clone().repeat(max(num_regions - 1, 1), 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        autocast_ctx = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda"
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
