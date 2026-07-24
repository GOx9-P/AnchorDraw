from __future__ import annotations

from typing import Sequence

from PIL import Image
import torch

from .image_ops import make_masked_background_image, make_masked_foreground_crop


class CLIPScoreAccumulator:
    def __init__(
        self,
        device: torch.device,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 32,
    ):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required for CLIP metrics. "
                "Install it with: pip install open-clip-torch"
            ) from exc

        self.device = device
        self.batch_size = int(batch_size)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        self._pg_sum = 0.0
        self._pg_count = 0
        self._bg_sum = 0.0
        self._bg_count = 0
        self._fg_sum = 0.0
        self._fg_count = 0

    @torch.no_grad()
    def _score_pairs(self, images: Sequence[Image.Image], texts: Sequence[str]) -> list[float]:
        if len(images) != len(texts):
            raise ValueError("images and texts must have the same length")

        scores: list[float] = []
        for start in range(0, len(images), self.batch_size):
            image_chunk = images[start:start + self.batch_size]
            text_chunk = texts[start:start + self.batch_size]
            image_batch = torch.stack([self.preprocess(image) for image in image_chunk], dim=0).to(self.device)
            text_batch = self.tokenizer(list(text_chunk)).to(self.device)

            image_features = self.model.encode_image(image_batch)
            text_features = self.model.encode_text(text_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pair_scores = (image_features * text_features).sum(dim=-1)
            scores.extend(float(value) for value in pair_scores.detach().cpu())
        return scores

    def update_prompt_global(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> None:
        scores = self._score_pairs(images, prompts)
        self._pg_sum += sum(scores)
        self._pg_count += len(scores)

    def update_background_region(
        self,
        generated_image: Image.Image,
        foreground_masks: torch.Tensor,
        background_prompt: str,
        apply_mask: bool = True,
    ) -> None:
        background_image = make_masked_background_image(
            generated_image,
            foreground_masks,
            apply_mask=apply_mask,
        )
        scores = self._score_pairs([background_image], [background_prompt])
        self._bg_sum += sum(scores)
        self._bg_count += len(scores)

    def update_foreground_regions(
        self,
        generated_image: Image.Image,
        masks: torch.Tensor,
        prompts: Sequence[str],
        padding_ratio: float = 0.08,
        apply_mask: bool = True,
    ) -> None:
        if len(prompts) == 0:
            return
        crops = [
            make_masked_foreground_crop(
                generated_image,
                masks[index],
                padding_ratio=padding_ratio,
                apply_mask=apply_mask,
            )
            for index in range(len(prompts))
        ]
        scores = self._score_pairs(crops, list(prompts))
        self._fg_sum += sum(scores)
        self._fg_count += len(scores)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self._pg_count:
            value = self._pg_sum / self._pg_count
            metrics["clip_pg"] = value
            metrics["clip_pg_x100"] = value * 100.0
            metrics["clip_pg_count"] = float(self._pg_count)
        if self._bg_count:
            value = self._bg_sum / self._bg_count
            metrics["clip_bg"] = value
            metrics["clip_bg_x100"] = value * 100.0
            metrics["clip_bg_count"] = float(self._bg_count)
        if self._fg_count:
            value = self._fg_sum / self._fg_count
            metrics["clip_fg"] = value
            metrics["clip_fg_x100"] = value * 100.0
            metrics["clip_fg_count"] = float(self._fg_count)
        return metrics
