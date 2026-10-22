"""SD3 joint-attention capture and delayed AnchorDraw step control.

This adapter is specific to the pinned Flash-SD3 diffusers commit used by the
SD3 notebooks. The original joint-attention processor still computes the
transformer output; the extra attention call only measures image-to-text mass.
"""

from __future__ import annotations

from dataclasses import asdict
from math import isqrt
from typing import Sequence

import torch
import torch.nn.functional as F

from experiments.semantic_anchor import (
    _mask_bbox_center_shifts,
    _reference_point_shifts,
    build_weighted_overlap_masks,
    compute_anchor_measurements,
)


def _strict_token_positions(tokenizer, prompt: str, target: str, max_length: int) -> list[int]:
    """Find the category token span without falling back to unrelated words."""
    prompt_ids = tokenizer(
        prompt, padding="max_length", max_length=max_length,
        truncation=True,
    )["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    for start in range(len(prompt_ids) - len(target_ids) + 1):
        if prompt_ids[start : start + len(target_ids)] == target_ids:
            return list(range(start, start + len(target_ids)))
    return []


def sd3_category_token_positions(pipe, prompts: Sequence[str], categories: Sequence[str]) -> list[list[int]]:
    """Map category text to the shared CLIP token positions of SD3's prompt."""
    if len(prompts) != len(categories):
        raise ValueError("One category is required for each foreground prompt.")
    positions: list[list[int]] = [[]]
    for prompt, category in zip(prompts, categories):
        first = _strict_token_positions(pipe.tokenizer, prompt, category, pipe.tokenizer_max_length)
        second = _strict_token_positions(pipe.tokenizer_2, prompt, category, pipe.tokenizer_max_length)
        if not first or not second:
            raise ValueError(f"Category tokens missing from an SD3 CLIP prompt: {category!r}, {prompt!r}")
        # SD3 concatenates the two CLIP embeddings channelwise at each token
        # index. A union of different spans would mix category and other words.
        if first != second:
            raise ValueError(
                f"SD3 CLIP token spans disagree for {category!r}: "
                f"tokenizer={first}, tokenizer_2={second}; prompt={prompt!r}"
            )
        if max(first) >= pipe.tokenizer_max_length:
            raise ValueError("Category token index exceeds SD3's CLIP sequence length.")
        positions.append(first)
    return positions


class _JointAttentionProbe:
    def __init__(self, owner: "SD3AnchorController", original) -> None:
        self.owner = owner
        self.original = original

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, *args, **kwargs,
    ):
        result = self.original(
            attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
        )
        owner = self.owner
        if owner.current_step is None:
            return result
        if attention_mask is not None or args or kwargs:
            raise RuntimeError("Unexpected SD3 joint-attention arguments; inspect pinned diffusers API.")
        if hidden_states.ndim != 3 or encoder_hidden_states is None or encoder_hidden_states.ndim != 3:
            raise RuntimeError("Expected SD3 image and context token sequences.")

        batch, image_tokens, _ = hidden_states.shape
        context_tokens = encoder_hidden_states.shape[1]
        side = isqrt(image_tokens)
        if batch != owner.num_regions or side * side != image_tokens:
            raise RuntimeError(f"Unexpected SD3 attention shape: {tuple(hidden_states.shape)}")
        if owner.current_step in owner.maps:
            raise RuntimeError("Selected SD3 attention layer ran more than once in a step.")

        # JointAttnProcessor2_0 concatenates image keys before context keys.
        # A marker value of one at target context tokens makes SDPA return the
        # exact summed attention probability to those tokens, without forming
        # the quadratic attention matrix in Python.
        with torch.no_grad():
            query = attn.to_q(hidden_states)
            image_keys = attn.to_k(hidden_states)
            context_keys = attn.add_k_proj(encoder_hidden_states)
            heads = int(attn.heads)
            head_dim = query.shape[-1] // heads
            query = query.view(batch, image_tokens, heads, head_dim).transpose(1, 2)
            image_keys = image_keys.view(batch, image_tokens, heads, head_dim).transpose(1, 2)
            context_keys = context_keys.view(batch, context_tokens, heads, head_dim).transpose(1, 2)
            if getattr(attn, "norm_q", None) is not None:
                query = attn.norm_q(query)
            if getattr(attn, "norm_k", None) is not None:
                image_keys = attn.norm_k(image_keys)
            if getattr(attn, "norm_added_k", None) is not None:
                context_keys = attn.norm_added_k(context_keys)
            keys = torch.cat([image_keys, context_keys], dim=2)
            # The value dimension is one: SDPA returns only the target-token
            # probability mass, without a redundant head_dim-wide value copy.
            marker = keys.new_zeros(batch, heads, image_tokens + context_tokens, 1)
            for region, positions in enumerate(owner.token_positions):
                if region == 0:
                    continue
                valid = [index for index in positions if 0 <= index < context_tokens]
                if not valid:
                    raise RuntimeError(f"No valid SD3 target tokens for region {region}.")
                marker[region, :, [image_tokens + index for index in valid], 0] = 1
            mass = F.scaled_dot_product_attention(
                query, keys, marker, dropout_p=0.0, is_causal=False
            )[..., 0].mean(dim=1)
            maps = mass[1:].reshape(batch - 1, side, side).detach().float().cpu()
            if not torch.isfinite(maps).all():
                raise FloatingPointError("SD3 joint-attention map contains NaN/Inf.")
            if (maps < -1e-3).any() or (maps > 1.0 + 1e-3).any():
                raise RuntimeError("SD3 category attention mass is outside [0, 1].")
            if (maps.flatten(1).amax(dim=1) <= 0).any():
                raise RuntimeError("SD3 category attention map is entirely zero.")
            owner.maps[owner.current_step] = maps
        return result


class SD3AnchorController:
    """Use previous-step SD3 attention and denoised latents from step 2 onward.

    ``noop`` instruments the same baseline path for parity testing. ``anchor``
    uses the top-10%-in-mask reference and adaptive bilateral overlap weights.
    A full 1024x1024 image is one latent view; tiled canvases are rejected by
    the pipeline hook.
    """

    def __init__(
        self, pipeline, *, foreground_prompts: Sequence[str],
        category_names: Sequence[str], foreground_masks: torch.Tensor,
        mode: str = "anchor", attention_layer_index: int = 4,
        topk_percent: float = 10.0, spatial_sigma_latent: float = 4.0,
        semantic_sigma_scale: float = 1.0,
    ) -> None:
        if mode not in {"noop", "anchor"}:
            raise ValueError(f"Unknown SD3 controller mode: {mode}")
        if (not 0.0 < topk_percent <= 100.0 or spatial_sigma_latent <= 0
                or semantic_sigma_scale <= 0):
            raise ValueError("Invalid attention top-k percentage or weighting sigma.")
        self.pipeline = pipeline
        self.mode = mode
        self.layer_index = int(attention_layer_index)
        self.topk_percent = float(topk_percent)
        self.spatial_sigma_latent = float(spatial_sigma_latent)
        self.semantic_sigma_scale = float(semantic_sigma_scale)
        self.object_masks = foreground_masks.detach().float().cpu()
        if self.object_masks.ndim != 4 or self.object_masks.shape[1] != 1:
            raise ValueError("Foreground masks must be [regions,1,height,width].")
        if len(foreground_prompts) != self.object_masks.shape[0]:
            raise ValueError("Foreground prompts and masks have different counts.")
        self.token_positions = sd3_category_token_positions(
            pipeline.pipe, foreground_prompts, category_names
        )
        self.num_regions = self.object_masks.shape[0] + 1
        self.maps: dict[int, torch.Tensor] = {}
        self.records: list[dict] = []
        self.current_step: int | None = None
        self.current_shifts: list[tuple[int, int]] = []
        self.previous_shifts: list[tuple[int, int]] | None = None
        self.previous_features: torch.Tensor | None = None
        self.previous_anchors: list[tuple[float, float]] | None = None
        self.recorded_steps: set[int] = set()
        self._transformer = None
        self._original_processors = None

    def __enter__(self):
        transformer = self.pipeline.transformer
        if hasattr(transformer, "get_base_model"):
            transformer = transformer.get_base_model()
        if not hasattr(transformer, "attn_processors") or not hasattr(transformer, "set_attn_processor"):
            raise RuntimeError("Cannot access SD3 transformer attention processors.")
        original = dict(transformer.attn_processors)
        name = f"transformer_blocks.{self.layer_index}.attn.processor"
        if name not in original:
            raise RuntimeError(f"SD3 attention layer not found: {name}; found {list(original)[:5]}")
        processor = original[name]
        if type(processor).__name__ != "JointAttnProcessor2_0":
            raise RuntimeError(f"Expected JointAttnProcessor2_0 at {name}, got {type(processor).__name__}.")
        modified = dict(original)
        modified[name] = _JointAttentionProbe(self, processor)
        transformer.set_attn_processor(modified)
        self._transformer = transformer
        self._original_processors = original
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._original_processors is not None:
            self._transformer.set_attn_processor(self._original_processors)
        self._original_processors = None
        self._transformer = None

    def begin(self, *, height: int, width: int, num_regions: int,
              bootstrap_steps: int, masks: torch.Tensor) -> None:
        if (num_regions != self.num_regions or masks.shape[0] != num_regions
                or tuple(self.object_masks.shape[-2:]) != (height, width)):
            raise RuntimeError("SD3 controller inputs do not match the baseline masks.")
        if bootstrap_steps < 1 or bootstrap_steps >= masks.shape[1]:
            raise ValueError("SD3 controller needs at least one bootstrap and one later step.")
        self.height, self.width = height, width
        self.latent_size = tuple(masks.shape[-2:])
        self.expected_steps = masks.shape[1]
        self.bootstrap_steps = bootstrap_steps
        self.maps.clear()
        self.records.clear()
        self.recorded_steps.clear()
        self.previous_shifts = None
        self.previous_features = None
        self.previous_anchors = None
        self.current_step = None

    def _anchors_from_previous(self, step: int) -> list[tuple[float, float]]:
        if step not in self.maps or self.previous_shifts is None:
            raise RuntimeError(f"Missing previous-step SD3 attention or shifts at step {step}.")
        captured = self.maps[step]
        if captured.shape[0] != self.num_regions - 1:
            raise RuntimeError("Captured SD3 region count does not match masks.")
        anchors = []
        for region, native_map in enumerate(captured):
            latent_map = F.interpolate(
                native_map[None, None], size=self.latent_size,
                mode="bilinear", align_corners=False,
            )[0, 0]
            dy, dx = self.previous_shifts[region + 1]
            latent_map = latent_map.roll(shifts=(-dy, -dx), dims=(-2, -1))
            image_map = F.interpolate(
                latent_map[None, None], size=(self.height, self.width),
                mode="bilinear", align_corners=False,
            )[0, 0]
            measurement = compute_anchor_measurements(
                image_map, self.object_masks[region], topk_percent=self.topk_percent
            )
            if not measurement["topk_anchor_inside_mask"]:
                raise RuntimeError(f"SD3 anchor escaped foreground mask {region}.")
            anchors.append((measurement["topk_anchor_x"], measurement["topk_anchor_y"]))
        return anchors

    def prepare_step(self, step: int, timestep, raw_masks: torch.Tensor) -> torch.Tensor:
        if self.recorded_steps != set(range(step)):
            raise RuntimeError(f"Unexpected SD3 step order: {step}.")
        if step > 0:
            self.previous_anchors = self._anchors_from_previous(step - 1)
        self.current_step = step
        if step < self.bootstrap_steps or self.mode == "noop":
            self.records.append({"step": step, "mode": "baseline", "anchors": self.previous_anchors})
            return raw_masks
        if self.previous_anchors is None or self.previous_features is None:
            raise RuntimeError("SD3 weighting needs preceding attention and denoised region latents.")
        weighted, diagnostic = build_weighted_overlap_masks(
            raw_masks, self.previous_features, self.previous_anchors,
            image_size=(self.height, self.width), policy="adaptive_bilateral",
            spatial_sigma_latent=self.spatial_sigma_latent,
            semantic_sigma_scale=self.semantic_sigma_scale,
        )
        if weighted.shape != raw_masks.shape or not torch.isfinite(weighted).all():
            raise RuntimeError("SD3 weighted masks have invalid shape or values.")
        diagnostic.step_index = step
        diagnostic.timestep = int(timestep.item())
        self.records.append({
            "step": step, "mode": "anchor", "anchors": self.previous_anchors,
            "weighting": asdict(diagnostic),
        })
        return weighted

    def center_latents(self, step: int, latents: torch.Tensor,
                       masks: torch.Tensor) -> torch.Tensor:
        if step < self.bootstrap_steps:
            shifts = _mask_bbox_center_shifts(masks)
        elif self.mode == "anchor":
            shifts = [(0, 0)] + _reference_point_shifts(
                self.previous_anchors, image_size=(self.height, self.width),
                latent_size=latents.shape[-2:],
            )
        else:
            shifts = [(0, 0)] * self.num_regions
        if len(shifts) != latents.shape[0]:
            raise RuntimeError("SD3 centering shifts do not match regions.")
        self.current_shifts = shifts
        return torch.stack([
            latent.roll(shifts=shift, dims=(-2, -1))
            for latent, shift in zip(latents, shifts)
        ])

    def uncenter_latents(self, step: int, latents: torch.Tensor) -> torch.Tensor:
        if len(self.current_shifts) != latents.shape[0]:
            raise RuntimeError(f"Missing SD3 centering shift at step {step}.")
        return torch.stack([
            latent.roll(shifts=(-dy, -dx), dims=(-2, -1))
            for latent, (dy, dx) in zip(latents, self.current_shifts)
        ])

    def record_denoised(self, step: int, timestep, latents: torch.Tensor,
                        masks: torch.Tensor) -> None:
        if step in self.recorded_steps or self.current_step != step:
            raise RuntimeError("SD3 controller supports one latent view and one record per step.")
        if latents.shape[0] != self.num_regions or not torch.isfinite(latents).all():
            raise FloatingPointError("SD3 denoised region latents are nonfinite or mismatched.")
        self.previous_features = latents[1:].detach().clone()
        self.previous_shifts = list(self.current_shifts)
        self.recorded_steps.add(step)

    def finish(self) -> None:
        expected = set(range(self.expected_steps))
        if self.recorded_steps != expected or set(self.maps) != expected:
            raise RuntimeError(
                f"Incomplete SD3 controller: latents={sorted(self.recorded_steps)}, "
                f"attention={sorted(self.maps)}, expected={sorted(expected)}"
            )
