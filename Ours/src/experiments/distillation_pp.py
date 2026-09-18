"""Inference-time Distillation++ refinement for AnchorDraw.

This module implements the interpolative denoising update in Eq. (9) and
Algorithm 1 of "Inference-Time Diffusion Model Distillation" (Park et al.).
It is deliberately independent of region fusion: the caller refines each
region's student estimate first, then AnchorDraw performs weighted masking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import List

import torch


@dataclass
class DistillationPPStep:
    """Diagnostics for one Distillation++ decision in the student loop."""

    step_index: int
    timestep: int
    teacher_timestep: int | None
    applied: bool
    teacher_guidance_scale: float
    teacher_cfg_scale: float
    renoise_strategy: str
    teacher_elapsed_sec: float
    student_teacher_l2: float | None
    refinement_l2: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class DistillationPPRefiner:
    """Apply teacher-guided refinement to per-region clean estimates.

    For the first ``guide_steps`` student iterations:

    1. Re-noise the student's clean estimate to the next student timestep
       ``s = t - delta_t`` using fresh Gaussian noise.
    2. Denoise the perturbed estimate with the non-distilled teacher UNet.
    3. Interpolate ``x0_new = (1-lambda) * x0_student + lambda * x0_teacher``.

    A private RNG keeps the student's scheduler-noise stream identical across
    sweep branches.  This changes no distributional assumption of the paper;
    it only makes paired ablations reproducible.  ``lambda=0`` performs no
    teacher call and consumes no random number, preserving the W03 baseline.
    """

    def __init__(
        self,
        *,
        teacher_unet,
        pipeline,
        teacher_guidance_scale: float,
        guide_steps: int = 1,
        teacher_cfg_scale: float = 7.5,
        renoise_strategy: str = "random_next_timestep",
        seed_offset: int = 1_000_003,
    ) -> None:
        if not 0.0 <= teacher_guidance_scale <= 1.0:
            raise ValueError("teacher_guidance_scale (lambda) must be in [0, 1].")
        if guide_steps < 0:
            raise ValueError("guide_steps must be non-negative.")
        if teacher_cfg_scale < 0:
            raise ValueError("teacher_cfg_scale must be non-negative.")
        if renoise_strategy != "random_next_timestep":
            raise ValueError("Only the paper-aligned random_next_timestep strategy is supported.")

        self.teacher_unet = teacher_unet
        self.pipeline = pipeline
        self.teacher_guidance_scale = float(teacher_guidance_scale)
        self.guide_steps = int(guide_steps)
        self.teacher_cfg_scale = float(teacher_cfg_scale)
        self.renoise_strategy = renoise_strategy
        self.seed_offset = int(seed_offset)
        self.records: List[DistillationPPStep] = []
        self._generator: torch.Generator | None = None

        self.teacher_unet.eval()
        for parameter in self.teacher_unet.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _as_int(value) -> int:
        if isinstance(value, torch.Tensor):
            value = value.detach().flatten()[0].item()
        return int(value)

    def configure_sample(self, seed: int) -> None:
        """Reset diagnostics and create the private re-noising RNG."""

        device = torch.device(self.pipeline.device)
        generator_device = device.type if device.type == "cuda" else "cpu"
        self._generator = torch.Generator(device=generator_device)
        self._generator.manual_seed(int(seed) + self.seed_offset)
        self.records.clear()

    def _record_skipped(self, step_index: int, timestep) -> None:
        self.records.append(
            DistillationPPStep(
                step_index=int(step_index),
                timestep=self._as_int(timestep),
                teacher_timestep=None,
                applied=False,
                teacher_guidance_scale=self.teacher_guidance_scale,
                teacher_cfg_scale=self.teacher_cfg_scale,
                renoise_strategy=self.renoise_strategy,
                teacher_elapsed_sec=0.0,
                student_teacher_l2=None,
                refinement_l2=None,
            )
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        student_denoised: torch.Tensor,
        step_index: int,
        timestep,
        text_embeddings: torch.Tensor,
        student_guidance_scale: float,
    ) -> torch.Tensor:
        del student_guidance_scale  # Student CFG was already applied upstream.

        next_index = int(step_index) + 1
        should_apply = (
            self.teacher_guidance_scale > 0.0
            and int(step_index) < self.guide_steps
            and next_index < len(self.pipeline.timesteps)
        )
        if not should_apply:
            self._record_skipped(step_index, timestep)
            return student_denoised

        if self._generator is None:
            raise RuntimeError("Call configure_sample(seed) before generation.")
        region_count = int(student_denoised.shape[0])
        if int(text_embeddings.shape[0]) != 2 * region_count:
            raise ValueError(
                "Expected unconditional+conditional embeddings for every region: "
                f"got {text_embeddings.shape[0]} embeddings for {region_count} regions."
            )

        alpha_s = self.pipeline.alpha_prod_t_sqrt[next_index].to(
            device=student_denoised.device, dtype=student_denoised.dtype
        )
        sigma_s = self.pipeline.beta_prod_t_sqrt[next_index].to(
            device=student_denoised.device, dtype=student_denoised.dtype
        )
        teacher_timestep = self.pipeline.timesteps[next_index]
        renoise_noise = torch.randn(
            student_denoised.shape,
            generator=self._generator,
            device=student_denoised.device,
            dtype=student_denoised.dtype,
        )
        teacher_input = alpha_s * student_denoised + sigma_s * renoise_noise

        started = perf_counter()
        teacher_output = self.teacher_unet(
            torch.cat([teacher_input, teacher_input], dim=0),
            teacher_timestep,
            encoder_hidden_states=text_embeddings,
        )
        teacher_noise = teacher_output["sample"] if isinstance(teacher_output, dict) else teacher_output.sample
        teacher_noise_uncond, teacher_noise_cond = teacher_noise.chunk(2)
        teacher_noise_cfg = teacher_noise_uncond + self.teacher_cfg_scale * (
            teacher_noise_cond - teacher_noise_uncond
        )
        teacher_denoised = (teacher_input - sigma_s * teacher_noise_cfg) / alpha_s.clamp_min(1e-8)
        elapsed = perf_counter() - started

        refined = torch.lerp(
            student_denoised,
            teacher_denoised.to(dtype=student_denoised.dtype),
            self.teacher_guidance_scale,
        )
        if not torch.isfinite(refined).all():
            raise FloatingPointError("Distillation++ produced a non-finite refined latent.")

        student_teacher_l2 = float(
            (teacher_denoised.float() - student_denoised.float()).pow(2).mean().sqrt().item()
        )
        refinement_l2 = float(
            (refined.float() - student_denoised.float()).pow(2).mean().sqrt().item()
        )
        self.records.append(
            DistillationPPStep(
                step_index=int(step_index),
                timestep=self._as_int(timestep),
                teacher_timestep=self._as_int(teacher_timestep),
                applied=True,
                teacher_guidance_scale=self.teacher_guidance_scale,
                teacher_cfg_scale=self.teacher_cfg_scale,
                renoise_strategy=self.renoise_strategy,
                teacher_elapsed_sec=float(elapsed),
                student_teacher_l2=student_teacher_l2,
                refinement_l2=refinement_l2,
            )
        )
        return refined
