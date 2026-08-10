"""Cross-attention capture and Semantic Anchor measurements.

The capture utility temporarily replaces only the cross-attention processors
(`attn2`) in a Diffusers UNet. The denoising loop and scheduler remain owned by
the original SemanticDraw pipeline. Captured maps are reduced online across
attention heads and target tokens to keep the smoke experiment memory bounded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isqrt
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F


def find_target_token_indices(tokenizer, prompt: str, target_text: str) -> List[int]:
    """Locate target-text tokens inside a foreground prompt.

    COCO prompts normally follow ``a {category}``, so this first searches for
    the category token sequence. If tokenization differs because of whitespace
    handling, it falls back to every non-special token in the prompt and makes
    that fallback explicit in the exported debug metadata.
    """

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    if target_ids:
        width = len(target_ids)
        for start in range(0, len(prompt_ids) - width + 1):
            if prompt_ids[start : start + width] == target_ids:
                return list(range(start, start + width))

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    fallback = [index for index, token_id in enumerate(prompt_ids) if token_id not in special_ids]
    if not fallback:
        raise ValueError(f"Khong tim thay token noi dung trong prompt: {prompt!r}")
    return fallback


@dataclass
class CapturedLayerMap:
    layer_name: str
    spatial_size: int
    values: torch.Tensor


class _CaptureStore:
    def __init__(self) -> None:
        self.enabled = False
        self.current_timestep: int | None = None
        self.region_count = 0
        self.token_indices: List[List[int]] = []
        self.maps: MutableMapping[Tuple[int, int], List[CapturedLayerMap]] = defaultdict(list)

    def configure(self, token_indices: Sequence[Sequence[int]]) -> None:
        self.region_count = len(token_indices)
        self.token_indices = [list(indices) for indices in token_indices]
        self.maps.clear()
        self.current_timestep = None
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_timestep(self, timestep) -> None:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.detach().flatten()[0].item()
        self.current_timestep = int(timestep)

    def add(self, layer_name: str, attention_probs: torch.Tensor, batch_size: int, heads: int) -> None:
        if not self.enabled or self.current_timestep is None or self.region_count == 0:
            return
        if attention_probs.ndim != 3 or batch_size < self.region_count:
            return

        query_length = int(attention_probs.shape[1])
        side = isqrt(query_length)
        if side * side != query_length:
            return

        key_length = int(attention_probs.shape[2])
        probs = attention_probs.reshape(batch_size, heads, query_length, key_length)
        conditional_start = batch_size - self.region_count
        conditional = probs[conditional_start:]

        for region_index, indices in enumerate(self.token_indices):
            valid = [index for index in indices if 0 <= index < key_length]
            if not valid:
                continue
            selected = conditional[region_index].index_select(
                -1,
                torch.as_tensor(valid, device=conditional.device),
            )
            token_map = selected.mean(dim=(0, 2))
            token_map = token_map.reshape(side, side).detach().float().cpu()
            min_value = token_map.min()
            max_value = token_map.max()
            token_map = (token_map - min_value) / (max_value - min_value).clamp_min(1e-8)
            self.maps[(self.current_timestep, region_index)].append(
                CapturedLayerMap(layer_name=layer_name, spatial_size=side, values=token_map)
            )


class _CrossAttentionCaptureProcessor:
    """Diffusers AttnProcessor-compatible implementation with map capture."""

    def __init__(self, layer_name: str, store: _CaptureStore) -> None:
        self.layer_name = layer_name
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.store.add(self.layer_name, attention_probs, batch_size, int(attn.heads))

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


class SemanticAnchorCapture:
    """Context manager that captures target-token cross-attention maps."""

    def __init__(self, unet) -> None:
        self.unet = unet
        self.store = _CaptureStore()
        self._original_processors: Mapping[str, object] | None = None
        self._unet_hook = None

    def _on_unet_pre_forward(self, module, args, kwargs=None) -> None:
        timestep = None
        if kwargs and "timestep" in kwargs:
            timestep = kwargs["timestep"]
        elif len(args) > 1:
            timestep = args[1]
        if timestep is not None:
            self.store.set_timestep(timestep)

    def install(self) -> "SemanticAnchorCapture":
        if self._original_processors is not None:
            return self
        self._original_processors = dict(self.unet.attn_processors)
        processors = {}
        for name, processor in self._original_processors.items():
            if name.endswith("attn2.processor"):
                processors[name] = _CrossAttentionCaptureProcessor(name, self.store)
            else:
                processors[name] = processor
        self.unet.set_attn_processor(processors)
        try:
            self._unet_hook = self.unet.register_forward_pre_hook(self._on_unet_pre_forward, with_kwargs=True)
        except TypeError:
            self._unet_hook = self.unet.register_forward_pre_hook(self._on_unet_pre_forward)
        return self

    def restore(self) -> None:
        self.store.disable()
        if self._unet_hook is not None:
            self._unet_hook.remove()
            self._unet_hook = None
        if self._original_processors is not None:
            self.unet.set_attn_processor(dict(self._original_processors))
            self._original_processors = None

    def configure(self, token_indices: Sequence[Sequence[int]]) -> None:
        self.store.configure(token_indices)

    @property
    def maps(self) -> Mapping[Tuple[int, int], List[CapturedLayerMap]]:
        return self.store.maps

    def __enter__(self) -> "SemanticAnchorCapture":
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.restore()


@dataclass
class CapturedLatentStep:
    """Global canvas latent immediately after one reverse-diffusion step."""

    step_index: int
    timestep: int
    latent: torch.Tensor


class SemanticLatentStepCapture:
    """Capture SemanticDraw's merged canvas latent without changing its loop.

    SemanticDraw calls ``scheduler_add_noise`` after each merged denoising step,
    except for the final step. The wrapper stores that method's input before
    noise is added. The final latent is captured when the baseline calls
    ``decode_latents``. Captures are copied to CPU and can therefore be decoded
    after generation without retaining the UNet graph or GPU activations.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.enabled = False
        self.timesteps: List[int] = []
        self.records: List[CapturedLatentStep] = []
        self._next_noise_index = 1
        self._original_scheduler_add_noise = None
        self._original_decode_latents = None

    @staticmethod
    def _as_int(value) -> int:
        if isinstance(value, torch.Tensor):
            value = value.detach().flatten()[0].item()
        return int(value)

    def configure(self, timesteps: Sequence[int], enabled: bool = True) -> None:
        self.timesteps = [self._as_int(timestep) for timestep in timesteps]
        self.records.clear()
        self._next_noise_index = 1
        self.enabled = enabled

    def disable(self) -> None:
        self.enabled = False

    def _store(self, step_index: int, latent: torch.Tensor) -> None:
        if step_index < 0 or step_index >= len(self.timesteps):
            return
        if any(record.step_index == step_index for record in self.records):
            return
        self.records.append(
            CapturedLatentStep(
                step_index=step_index,
                timestep=self.timesteps[step_index],
                latent=latent.detach().to(device="cpu", dtype=torch.float16).clone(),
            )
        )

    def install(self) -> "SemanticLatentStepCapture":
        if self._original_scheduler_add_noise is not None:
            return self

        self._original_scheduler_add_noise = self.pipeline.scheduler_add_noise
        self._original_decode_latents = self.pipeline.decode_latents

        def scheduler_add_noise_wrapper(latent, noise, index):
            index_int = self._as_int(index)
            if self.enabled and index_int == self._next_noise_index:
                self._store(index_int - 1, latent)
                self._next_noise_index += 1
            return self._original_scheduler_add_noise(latent, noise, index)

        def decode_latents_wrapper(latents, vae=None):
            if self.enabled and self.timesteps:
                self._store(len(self.timesteps) - 1, latents)
            return self._original_decode_latents(latents, vae)

        self.pipeline.scheduler_add_noise = scheduler_add_noise_wrapper
        self.pipeline.decode_latents = decode_latents_wrapper
        return self

    def restore(self) -> None:
        self.enabled = False
        if self._original_scheduler_add_noise is not None:
            self.pipeline.scheduler_add_noise = self._original_scheduler_add_noise
            self._original_scheduler_add_noise = None
        if self._original_decode_latents is not None:
            self.pipeline.decode_latents = self._original_decode_latents
            self._original_decode_latents = None

    def __enter__(self) -> "SemanticLatentStepCapture":
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.restore()


def aggregate_attention_maps(
    captured_maps: Mapping[Tuple[int, int], Iterable[CapturedLayerMap]],
    output_size: Tuple[int, int],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Resize and average normalized maps from all captured cross-attention layers."""

    aggregated: Dict[Tuple[int, int], torch.Tensor] = {}
    for key, layer_maps in captured_maps.items():
        resized = []
        for layer_map in layer_maps:
            values = layer_map.values[None, None]
            values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)[0, 0]
            resized.append(values)
        if resized:
            result = torch.stack(resized).mean(dim=0)
            result = (result - result.min()) / (result.max() - result.min()).clamp_min(1e-8)
            aggregated[key] = result
    return aggregated


def _mask_geometry(mask: torch.Tensor) -> Dict[str, float]:
    mask = mask.squeeze().bool().cpu()
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        raise ValueError("Mask rong, khong the tinh Semantic Anchor.")
    return {
        "centroid_x": float(xs.float().mean()),
        "centroid_y": float(ys.float().mean()),
        "bbox_center_x": float(xs.min() + xs.max()) / 2.0,
        "bbox_center_y": float(ys.min() + ys.max()) / 2.0,
    }


def compute_anchor_measurements(attention_map: torch.Tensor, mask: torch.Tensor) -> Dict[str, float | bool]:
    """Compute proposal-faithful ``argmax`` anchor inside a binary mask."""

    attention_map = attention_map.detach().float().cpu()
    mask = mask.squeeze().bool().cpu()
    if attention_map.shape != mask.shape:
        mask = F.interpolate(mask[None, None].float(), size=attention_map.shape, mode="nearest")[0, 0].bool()
    if not mask.any():
        raise ValueError("Mask rong sau khi resize.")

    masked_attention = attention_map.masked_fill(~mask, float("-inf"))
    anchor_flat = int(masked_attention.argmax())
    width = int(attention_map.shape[1])
    anchor_y, anchor_x = divmod(anchor_flat, width)

    global_flat = int(attention_map.argmax())
    global_y, global_x = divmod(global_flat, width)
    geometry = _mask_geometry(mask)

    centroid_distance = ((anchor_x - geometry["centroid_x"]) ** 2 + (anchor_y - geometry["centroid_y"]) ** 2) ** 0.5
    bbox_distance = ((anchor_x - geometry["bbox_center_x"]) ** 2 + (anchor_y - geometry["bbox_center_y"]) ** 2) ** 0.5
    global_peak_distance = ((anchor_x - global_x) ** 2 + (anchor_y - global_y) ** 2) ** 0.5
    diagonal = (attention_map.shape[0] ** 2 + attention_map.shape[1] ** 2) ** 0.5

    return {
        "anchor_x": float(anchor_x),
        "anchor_y": float(anchor_y),
        "anchor_attention": float(attention_map[anchor_y, anchor_x]),
        "global_peak_x": float(global_x),
        "global_peak_y": float(global_y),
        "global_peak_inside_mask": bool(mask[global_y, global_x]),
        "distance_to_global_peak_px": float(global_peak_distance),
        "distance_to_centroid_px": float(centroid_distance),
        "distance_to_bbox_center_px": float(bbox_distance),
        "distance_to_centroid_norm": float(centroid_distance / diagonal),
        "distance_to_bbox_center_norm": float(bbox_distance / diagonal),
        **geometry,
    }
