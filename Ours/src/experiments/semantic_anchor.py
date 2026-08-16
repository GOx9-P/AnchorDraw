"""Cross-attention capture and Semantic Anchor measurements.

The capture utility temporarily replaces only the cross-attention processors
(`attn2`) in a Diffusers UNet. The denoising loop and scheduler remain owned by
the original SemanticDraw pipeline. Captured maps are reduced online across
attention heads and target tokens to keep the smoke experiment memory bounded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil, isqrt
from typing import Dict, Iterable, List, Literal, Mapping, MutableMapping, Sequence, Tuple

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


@dataclass
class SemanticAnchorRuntimeStep:
    """How one denoising step was spatially centered at runtime.

    ``anchor_source_step_index`` is deliberately separate from ``step_index``:
    an attention map becomes available only *after* the UNet call of its own
    step, so it can guide the following denoising step, never the same one.
    """

    step_index: int
    timestep: int
    selection_source: str
    anchor_source_step_index: int | None
    anchor_source_timestep: int | None
    anchor_strategy: str
    points_xy: List[Tuple[float, float]]


def _shift_to_reference_points(
    latents: torch.Tensor,
    points_xy: Sequence[Tuple[float, float]],
    *,
    image_size: Tuple[int, int],
    reverse: bool = False,
) -> torch.Tensor:
    """Move per-region reference points to the latent canvas center.

    SemanticDraw's baseline moves each mask bounding-box center to the canvas
    center during bootstrap. This is the same operation with an arbitrary
    image-space point, then it is reversed after the UNet prediction.
    """

    if len(points_xy) != latents.shape[0]:
        raise ValueError(
            f"Expected {latents.shape[0]} reference points, got {len(points_xy)}."
        )

    image_h, image_w = image_size
    latent_h, latent_w = latents.shape[-2:]
    shifts: List[Tuple[int, int]] = []
    for x, y in points_xy:
        # Pixel centers are mapped consistently from image coordinates to
        # the VAE latent canvas. ``roll`` expects (dy, dx).
        x_latent = round(float(x) * latent_w / image_w)
        y_latent = round(float(y) * latent_h / image_h)
        dy = latent_h // 2 - y_latent
        dx = latent_w // 2 - x_latent
        if reverse:
            dy, dx = -dy, -dx
        shifts.append((dy, dx))

    return torch.stack(
        [latent.roll(shifts=shift, dims=(-2, -1)) for latent, shift in zip(latents, shifts)],
        dim=0,
    )


class SemanticAnchorRuntime:
    """Runtime ablation of SemanticDraw's centering rule.

    The original baseline file is never changed.  This runner faithfully
    reproduces its SD1.5 denoising/mask-mixing loop for the experiment's input
    protocol, with one controlled intervention:

    * step 0 uses the baseline bbox-centering bootstrap exactly;
    * later steps either use no extra centering (``baseline``), bbox centers
      again (``bbox_control``), or the masked attention anchor captured at the
      immediately preceding step (``semantic_anchor``), or the attention-
      projected centroid of its strongest in-mask pixels
      (``semantic_topk_anchor``).

    The delayed use of attention is causally valid: the map from step ``i`` is
    produced by that step's UNet, therefore it can first be used at ``i + 1``.
    The final step is still executed and decoded normally.
    """

    def __init__(
        self,
        pipeline,
        attention_capture: SemanticAnchorCapture,
        *,
        image_size: Tuple[int, int],
    ) -> None:
        self.pipeline = pipeline
        self.attention_capture = attention_capture
        self.image_size = image_size
        self.step_records: List[SemanticAnchorRuntimeStep] = []

    def _anchors_from_current_step(
        self,
        timestep: int,
        foreground_masks: torch.Tensor,
        *,
        strategy: Literal["argmax", "topk_projected_centroid"],
        topk_percent: float,
    ) -> List[Tuple[float, float]]:
        maps = aggregate_attention_maps(self.attention_capture.maps, self.image_size)
        anchors: List[Tuple[float, float]] = []
        for region_index, mask in enumerate(foreground_masks):
            key = (int(timestep), region_index)
            if key not in maps:
                raise RuntimeError(
                    f"Missing cross-attention map for timestep={timestep}, region={region_index}."
                )
            measurement = compute_anchor_measurements(
                maps[key], mask.cpu(), topk_percent=topk_percent
            )
            if strategy == "argmax":
                anchors.append(
                    (float(measurement["anchor_x"]), float(measurement["anchor_y"]))
                )
            else:
                anchors.append(
                    (
                        float(measurement["topk_anchor_x"]),
                        float(measurement["topk_anchor_y"]),
                    )
                )
        return anchors

    @torch.no_grad()
    def generate(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        masks: torch.Tensor,
        foreground_masks: torch.Tensor,
        mode: Literal["baseline", "bbox_control", "semantic_anchor", "semantic_topk_anchor"],
        bootstrap_steps: int = 1,
        topk_percent: float = 10.0,
        mask_stds: float | Sequence[float] | None = None,
        mask_strengths: float | Sequence[float] | None = None,
        preprocess_mask_cover_alpha: float | None = None,
        guidance_scale: float | None = None,
        use_boolean_mask: bool = True,
    ):
        """Generate one image using the controlled centering ablation.

        This intentionally supports the manifest protocol used by the
        experiment: a tensor of mutually aligned background+foreground masks
        and one prompt per mask.  Background image/BLIP branches are outside
        the ablation and remain available through the unmodified baseline.
        """

        if mode not in {"baseline", "bbox_control", "semantic_anchor", "semantic_topk_anchor"}:
            raise ValueError(f"Unknown mode: {mode}")
        if not 0.0 < topk_percent <= 100.0:
            raise ValueError("topk_percent must be in the interval (0, 100].")
        if len(prompts) != len(negative_prompts) or len(prompts) != int(masks.shape[0]):
            raise ValueError("The experiment requires exactly one prompt and negative prompt per mask.")
        if int(masks.shape[0]) != int(foreground_masks.shape[0]) + 1:
            raise ValueError("Expected one background mask followed by foreground masks.")
        if bootstrap_steps != 1:
            raise ValueError("This ablation is defined for one baseline bootstrap step only.")

        pipeline = self.pipeline
        height, width = self.image_size
        if height > 512 or width > 512:
            raise ValueError("Semantic Anchor SD1.5 ablation currently validates the 512x512 protocol only.")

        num_masks = len(prompts)
        if guidance_scale is None:
            guidance_scale = pipeline.default_guidance_scale
        if mask_stds is None:
            mask_stds = pipeline.default_mask_std
        if mask_strengths is None:
            mask_strengths = pipeline.default_mask_strength
        if preprocess_mask_cover_alpha is None:
            preprocess_mask_cover_alpha = pipeline.default_preprocess_mask_cover_alpha

        fg_masks, _, _ = pipeline.process_mask(
            masks,
            mask_strengths,
            mask_stds,
            height=height,
            width=width,
            use_boolean_mask=use_boolean_mask,
            timesteps=pipeline.timesteps,
            preprocess_mask_cover_alpha=preprocess_mask_cover_alpha,
        )
        # The manifest already provides a background region, so no external
        # background latent is mixed in this controlled experiment.
        bg_masks = (1 - fg_masks.sum(dim=0)).clip_(0, 1)
        if bool((bg_masks.sum() > 1e-4).item()):
            raise ValueError("Background mask must complete the canvas for this ablation.")

        # Keep the baseline RNG order: its white VAE latent is created before
        # the initial diffusion noise. VAE latent sampling consumes randomness,
        # so constructing it later would invalidate same-seed parity checks.
        white_bootstrap = pipeline.get_white_background(height, width)

        uncond_embeds, text_embeds = pipeline.get_text_embeds(prompts, negative_prompts)
        if uncond_embeds.shape[0] != num_masks or text_embeds.shape[0] != num_masks:
            raise RuntimeError("Text embedding batch does not match the mask batch.")
        text_embeds = torch.cat([uncond_embeds, text_embeds])

        latent_h = (height + pipeline.vae_scale_factor - 1) // pipeline.vae_scale_factor
        latent_w = (width + pipeline.vae_scale_factor - 1) // pipeline.vae_scale_factor
        latent = torch.randn(
            (1, pipeline.unet.config.in_channels, latent_h, latent_w),
            dtype=pipeline.dtype,
            device=pipeline.device,
        )
        # The 512x512 protocol produces precisely one view. Supporting tiled
        # canvases requires translating anchors into every local view and is
        # deliberately excluded rather than silently doing the wrong thing.
        views = [(0, latent_h, 0, latent_w)]
        tile_masks = latent.new_ones((1, 1, latent_h, latent_w))
        value = torch.zeros_like(latent)
        count_all = torch.zeros_like(latent)
        timesteps = [int(value.item()) for value in pipeline.timesteps.detach().cpu()]
        previous_anchors: List[Tuple[float, float]] | None = None
        self.step_records = []

        with torch.autocast("cuda"):
            for step_index, timestep in enumerate(pipeline.timesteps):
                fg_mask = fg_masks[:, step_index]
                value.zero_()
                count_all.zero_()

                if step_index == 0:
                    selection_source = "baseline_bbox_bootstrap"
                    selected_points: List[Tuple[float, float]] = []
                    anchor_source_step = None
                elif mode == "baseline":
                    selection_source = "none"
                    selected_points = []
                    anchor_source_step = None
                elif mode == "bbox_control":
                    selection_source = "bbox_center"
                    selected_points = []
                    anchor_source_step = None
                else:
                    if previous_anchors is None:
                        raise RuntimeError("No previous-step anchors are available for Semantic Anchor centering.")
                    selection_source = (
                        "semantic_topk_attention_previous_step"
                        if mode == "semantic_topk_anchor"
                        else "semantic_anchor_previous_step"
                    )
                    selected_points = previous_anchors
                    anchor_source_step = step_index - 1

                for view_index, (h_start, h_end, w_start, w_end) in enumerate(views):
                    fg_mask_view = fg_mask[..., h_start:h_end, w_start:w_end]
                    latent_view = latent[..., h_start:h_end, w_start:w_end].repeat(num_masks, 1, 1, 1)

                    if step_index == 0:
                        # Exact baseline bootstrap path, including white latent,
                        # bbox centering, reverse centering and leakage removal.
                        white = white_bootstrap[..., h_start:h_end, w_start:w_end]
                        bg_latent = latent_view[:1]
                        mix_ratio = min(
                            1,
                            # Keep the baseline attribute spelling: the original
                            # constructor exposes ``default_boostrap_mix_steps``.
                            max(0, pipeline.default_boostrap_mix_steps - step_index),
                        )
                        bg_latent = mix_ratio * white + (1.0 - mix_ratio) * bg_latent
                        bg_latent = pipeline.scheduler_add_noise(bg_latent, None, step_index)
                        latent_view = (1 - fg_mask_view) * bg_latent + fg_mask_view * latent_view
                        from util import shift_to_mask_bbox_center
                        latent_view = shift_to_mask_bbox_center(latent_view, fg_mask_view, reverse=True)
                    elif mode == "bbox_control":
                        from util import shift_to_mask_bbox_center
                        # The first entry is the background mask; only actual
                        # object regions receive repeated geometric centering.
                        foreground_latent = shift_to_mask_bbox_center(latent_view[1:], fg_mask_view[1:], reverse=True)
                        latent_view = torch.cat([latent_view[:1], foreground_latent], dim=0)
                    elif mode in {"semantic_anchor", "semantic_topk_anchor"}:
                        foreground_latent = _shift_to_reference_points(
                            latent_view[1:],
                            selected_points,
                            image_size=self.image_size,
                            reverse=False,
                        )
                        latent_view = torch.cat([latent_view[:1], foreground_latent], dim=0)

                    noise_pred = pipeline.unet(
                        torch.cat([latent_view] * 2), timestep, encoder_hidden_states=text_embeds
                    )["sample"]
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latent_view = pipeline.scheduler_step(noise_pred, step_index, latent_view)

                    if step_index == 0:
                        from util import shift_to_mask_bbox_center
                        latent_view = shift_to_mask_bbox_center(latent_view, fg_mask_view)
                        leak = (latent_view - bg_latent).pow(2).mean(dim=1, keepdim=True)
                        leak_sigmoid = torch.sigmoid(leak / pipeline.default_bootstrap_leak_sensitivity) * 2 - 1
                        fg_mask_view = fg_mask_view * leak_sigmoid
                    elif mode == "bbox_control":
                        from util import shift_to_mask_bbox_center
                        foreground_latent = shift_to_mask_bbox_center(latent_view[1:], fg_mask_view[1:])
                        latent_view = torch.cat([latent_view[:1], foreground_latent], dim=0)
                    elif mode in {"semantic_anchor", "semantic_topk_anchor"}:
                        foreground_latent = _shift_to_reference_points(
                            latent_view[1:],
                            selected_points,
                            image_size=self.image_size,
                            reverse=True,
                        )
                        latent_view = torch.cat([latent_view[:1], foreground_latent], dim=0)

                    fg_mask_view = fg_mask_view * tile_masks[:, view_index : view_index + 1, h_start:h_end, w_start:w_end]
                    value[..., h_start:h_end, w_start:w_end] += (fg_mask_view * latent_view).sum(dim=0, keepdim=True)
                    count_all[..., h_start:h_end, w_start:w_end] += fg_mask_view.sum(dim=0, keepdim=True)

                latent = torch.where(count_all > 0, value / count_all, value)

                # The map for this step is captured by the UNet above. It is
                # stored now and used only at the next iteration.
                current_anchors = self._anchors_from_current_step(
                    int(timestep.item()),
                    foreground_masks,
                    strategy=("topk_projected_centroid" if mode == "semantic_topk_anchor" else "argmax"),
                    topk_percent=topk_percent,
                )
                self.step_records.append(
                    SemanticAnchorRuntimeStep(
                        step_index=step_index,
                        timestep=timesteps[step_index],
                        selection_source=selection_source,
                        anchor_source_step_index=anchor_source_step,
                        anchor_source_timestep=(timesteps[anchor_source_step] if anchor_source_step is not None else None),
                        anchor_strategy=("topk_projected_centroid" if mode == "semantic_topk_anchor" else "argmax"),
                        points_xy=selected_points,
                    )
                )
                previous_anchors = current_anchors

                if step_index < len(pipeline.timesteps) - 1:
                    latent = pipeline.scheduler_add_noise(latent, None, step_index + 1)

        image = pipeline.decode_latents(latent.to(dtype=pipeline.dtype))[0]
        from torchvision import transforms as T
        return T.ToPILImage()(image), list(self.step_records)


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


def compute_anchor_measurements(
    attention_map: torch.Tensor,
    mask: torch.Tensor,
    *,
    topk_percent: float = 10.0,
) -> Dict[str, float | bool]:
    """Measure both the masked argmax and a robust top-k attention anchor.

    The top-k anchor is derived from the centroid of the highest-scoring
    ``topk_percent`` pixels *inside the object mask*. That continuous center
    is projected back to its nearest selected pixel before being used as an
    anchor. The projection matters when separate high-attention islands have
    a mathematical center in the background.
    """

    attention_map = attention_map.detach().float().cpu()
    mask = mask.squeeze().bool().cpu()
    if attention_map.shape != mask.shape:
        mask = F.interpolate(mask[None, None].float(), size=attention_map.shape, mode="nearest")[0, 0].bool()
    if not mask.any():
        raise ValueError("Mask rong sau khi resize.")
    if not 0.0 < topk_percent <= 100.0:
        raise ValueError("topk_percent must be in the interval (0, 100].")

    masked_attention = attention_map.masked_fill(~mask, float("-inf"))
    anchor_flat = int(masked_attention.argmax())
    width = int(attention_map.shape[1])
    anchor_y, anchor_x = divmod(anchor_flat, width)

    mask_ys, mask_xs = torch.where(mask)
    in_mask_attention = attention_map[mask]
    topk_count = max(1, ceil(in_mask_attention.numel() * topk_percent / 100.0))
    topk_values, topk_indices = torch.topk(in_mask_attention, k=topk_count)
    topk_xs = mask_xs[topk_indices].float()
    topk_ys = mask_ys[topk_indices].float()
    # This is deliberately an unweighted centroid. The top-k operation has
    # already selected the semantic region; weighting again would make a
    # slightly higher single pixel dominate and collapse toward argmax.
    topk_center_x = float(topk_xs.mean())
    topk_center_y = float(topk_ys.mean())
    nearest_topk_pixel = int(
        ((topk_xs - topk_center_x).pow(2) + (topk_ys - topk_center_y).pow(2)).argmin()
    )
    # A point used to move a latent must correspond to a valid object location.
    # The weighted center itself can lie between disconnected attention islands.
    topk_anchor_x = float(topk_xs[nearest_topk_pixel])
    topk_anchor_y = float(topk_ys[nearest_topk_pixel])

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
        "topk_percent": float(topk_percent),
        "topk_pixel_count": int(topk_count),
        "topk_attention_threshold": float(topk_values.min()),
        "topk_center_x": topk_center_x,
        "topk_center_y": topk_center_y,
        "topk_anchor_x": topk_anchor_x,
        "topk_anchor_y": topk_anchor_y,
        "topk_anchor_inside_mask": bool(mask[int(topk_anchor_y), int(topk_anchor_x)]),
        "topk_anchor_attention": float(attention_map[int(topk_anchor_y), int(topk_anchor_x)]),
        "topk_anchor_attention_mean": float(topk_values.mean()),
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
