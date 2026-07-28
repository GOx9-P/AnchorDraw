from __future__ import annotations

import contextlib
from typing import Sequence

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm


def get_views(
    panorama_height: int,
    panorama_width: int,
    window_size: int = 64,
    stride: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Baseline MultiDiffusion sliding-window view list in latent coordinates."""
    latent_height = panorama_height / 8
    latent_width = panorama_width / 8
    num_blocks_height = (latent_height - window_size) // stride + 1
    num_blocks_width = (latent_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)

    views = []
    for index in range(total_num_blocks):
        h_start = int((index // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((index % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusionLCM:
    """Region-based MultiDiffusion fusion with SD1.5 LCM sampler.

    The mask/window fusion follows the original region-based MultiDiffusion
    implementation. The intentional change is replacing DDIM with the same
    SD1.5 LCM LoRA + LCMScheduler family used by SemanticDraw SD1.5 LCM.
    """

    def __init__(
        self,
        model_id: str,
        lcm_lora_id: str,
        lora_weight_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        t_index_list: Sequence[int] = (0, 4, 12, 25, 37),
        schedule_steps: int = 50,
        show_progress: bool = False,
    ) -> None:
        from diffusers import LCMScheduler, StableDiffusionPipeline

        self.model_id = model_id
        self.lcm_lora_id = lcm_lora_id
        self.device = device
        self.dtype = dtype
        self.t_index_list = list(t_index_list)
        self.schedule_steps = int(schedule_steps)
        self.show_progress = bool(show_progress)

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

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(
            lcm_lora_id,
            weight_name=lora_weight_name,
            adapter_name="lcm",
        )

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

        self.prepare_lcm_schedule(self.t_index_list, self.schedule_steps)

    @torch.no_grad()
    def prepare_lcm_schedule(
        self,
        t_index_list: Sequence[int],
        num_inference_steps: int,
    ) -> None:
        self.scheduler.set_timesteps(num_inference_steps)
        max_index = len(self.scheduler.timesteps) - 1
        for index in t_index_list:
            if index < 0 or index > max_index:
                raise ValueError(f"t_index {index} out of scheduler range 0..{max_index}")

        self.timesteps = torch.as_tensor(
            [int(self.scheduler.timesteps[index].item()) for index in t_index_list],
            dtype=torch.long,
            device=self.device,
        )

        shape = (len(t_index_list), 1, 1, 1)
        c_skip_list = []
        c_out_list = []
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []

        for timestep in self.timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)

        self.c_skip = torch.stack(c_skip_list).view(*shape).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(*shape).to(dtype=self.dtype, device=self.device)
        self.alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list).view(*shape).to(dtype=self.dtype, device=self.device)
        )
        self.beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list).view(*shape).to(dtype=self.dtype, device=self.device)
        )

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

    def scheduler_step(self, noise_pred: torch.Tensor, index: int, latent: torch.Tensor) -> torch.Tensor:
        """LCM denoise-only step without mutating scheduler-internal step index."""
        f_theta = (latent - self.beta_prod_t_sqrt[index] * noise_pred) / self.alpha_prod_t_sqrt[index]
        return self.c_out[index] * f_theta + self.c_skip[index] * latent

    def scheduler_add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor | None,
        index: int,
    ) -> torch.Tensor:
        if index >= len(self.alpha_prod_t_sqrt) or index < 0:
            return latent
        noise = torch.randn_like(latent) if noise is None else noise
        return self.alpha_prod_t_sqrt[index] * latent + self.beta_prod_t_sqrt[index] * noise

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
        if show_progress is None:
            show_progress = self.show_progress

        with autocast_ctx:
            for step_index, timestep in enumerate(
                tqdm(self.timesteps, leave=False, disable=not bool(show_progress))
            ):
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
                        background = self.scheduler_add_noise(background, background_noise, step_index)
                        latent_view[1:] = latent_view[1:] * masks_view[1:] + background * (1 - masks_view[1:])

                    latent_model_input = torch.cat([latent_view] * 2)
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeds,
                    )["sample"]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents_view_denoised = self.scheduler_step(noise_pred, step_index, latent_view)
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdim=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdim=True)

                latent = torch.where(count > 0, value / count, value)
                if step_index < len(self.timesteps) - 1:
                    latent = self.scheduler_add_noise(latent, None, step_index + 1)

        images = self.decode_latents(latent)
        return T.ToPILImage()(images[0].detach().cpu())
