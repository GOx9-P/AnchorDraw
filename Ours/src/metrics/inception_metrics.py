from __future__ import annotations

from typing import Tuple

from PIL import Image
import torch

from .image_ops import pil_list_to_uint8_tensor


class InceptionMetricsAccumulator:
    def __init__(
        self,
        device: torch.device,
        compute_fid: bool = True,
        compute_is: bool = True,
        fid_feature: int = 2048,
        is_splits: int = 10,
    ):
        self.device = device
        self.compute_fid = bool(compute_fid)
        self.compute_is = bool(compute_is)

        self.fid = None
        self.inception_score = None

        if self.compute_fid:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
            except ImportError as exc:
                raise ImportError(
                    "torchmetrics with torch-fidelity is required for FID. "
                    "Install it with: pip install torchmetrics torch-fidelity"
                ) from exc
            self.fid = FrechetInceptionDistance(feature=fid_feature, normalize=False).to(device)

        if self.compute_is:
            try:
                from torchmetrics.image.inception import InceptionScore
            except ImportError as exc:
                raise ImportError(
                    "torchmetrics is required for Inception Score. "
                    "Install it with: pip install torchmetrics torch-fidelity"
                ) from exc
            self.inception_score = InceptionScore(normalize=False, splits=int(is_splits)).to(device)

    def update(
        self,
        real_images: list[Image.Image],
        generated_images: list[Image.Image],
        target_size: Tuple[int, int],
    ) -> None:
        if not generated_images:
            return

        generated = pil_list_to_uint8_tensor(generated_images, target_size).to(self.device)
        if self.inception_score is not None:
            self.inception_score.update(generated)

        if self.fid is not None:
            real = pil_list_to_uint8_tensor(real_images, target_size).to(self.device)
            self.fid.update(real, real=True)
            self.fid.update(generated, real=False)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.fid is not None:
            metrics["fid"] = float(self.fid.compute().detach().cpu().item())
        if self.inception_score is not None:
            mean, std = self.inception_score.compute()
            metrics["is_mean"] = float(mean.detach().cpu().item())
            metrics["is_std"] = float(std.detach().cpu().item())
        return metrics
