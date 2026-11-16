"""Pluggable single-image metric depth for Uni3C rendering.

MoGe-3: https://github.com/microsoft/MoGe
Weights: Ruicheng/moge-3-vitl and Ruicheng/moge-3-vitg (MIT model cards).
The upstream package is imported lazily so the original Depth-Pro path remains
usable without installing MoGe's CUDA/sparse-refinement dependencies.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DepthPrediction:
    """RGB-aligned camera-z depth and pixel pinhole intrinsics."""
    depth: torch.Tensor  # [H,W], positive and finite
    intrinsic: torch.Tensor  # [3,3], current image pixels
    valid_mask: torch.Tensor  # [H,W], bool; renderer may use this later
    model_name: str
    filled_pixels: int


def create_depth_estimator(name="moge3-vitl", device="cuda", checkpoint=None,
                           refine_steps=3, use_fp16=False):
    """Return an estimator with ``predict(rgb_float_chw) -> DepthPrediction``.

    MoGe is selected explicitly by cam_render.py. Its pip installation is left
    to the user/environment because FlexGEMM/Triton needs a compatible toolchain.
    """
    if name == "depth-pro":
        import third_party.depth_pro.depth_pro as depth_pro
        model, transform = depth_pro.create_model_and_transforms(device=device)
        return DepthProEstimator(model.eval(), transform, device)
    variants = {
        "moge3-vitl": "Ruicheng/moge-3-vitl",
        "moge3-vitg": "Ruicheng/moge-3-vitg",
        "moge2-vitl": "Ruicheng/moge-2-vitl",
    }
    if name not in variants:
        raise ValueError(f"Unknown depth model {name!r}; choose from {['depth-pro', *variants]}.")
    try:
        if name.startswith("moge3"):
            from moge.model.v3 import MoGeModel
        else:
            from moge.model.v2 import MoGeModel
    except ImportError as error:
        raise ImportError(
            "MoGe is not installed. Follow docs/depth_estimation.md to install "
            "microsoft/MoGe and its FlexGEMM/Triton dependencies in the Uni3C environment."
        ) from error
    model = MoGeModel.from_pretrained(checkpoint or variants[name]).to(device).eval()
    return MoGeEstimator(model, name, device, refine_steps, use_fp16)


class DepthProEstimator:
    def __init__(self, model, transform, device):
        self.model, self.transform, self.device = model, transform, device

    @torch.inference_mode()
    def predict(self, image):
        # Depth-Pro's transform owns normalization and its 1536px inference.
        # The official transform expects H,W,C NumPy RGB. The common adapter
        # interface uses C,H,W float RGB in [0,1].
        image_np = (image.detach().float().cpu().permute(1, 2, 0).numpy()
                    .clip(0, 1) * 255).round().astype(np.uint8)
        prediction = self.model.infer(self.transform(image_np), f_px=None)
        depth = prediction["depth"].float().detach().cpu()
        focal = prediction["focallength_px"].item()
        height, width = depth.shape
        intrinsic = torch.tensor([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
        depth, mask, filled = repair_depth(depth)
        return DepthPrediction(depth, intrinsic, mask, "apple/DepthPro", filled)


class MoGeEstimator:
    def __init__(self, model, name, device, refine_steps=3, use_fp16=False):
        self.model, self.name, self.device = model, name, device
        self.refine_steps, self.use_fp16 = refine_steps, use_fp16

    @torch.inference_mode()
    def predict(self, image):
        """Infer RGB-aligned maps using MoGe's official token/refinement settings.

        MoGe internally decodes at its token-controlled resolution and resamples
        output maps to the input H,W. It is not native per-pixel prediction.
        Normalized intrinsics become pixel units after inference. No image scale
        is ever applied to metric z depth.
        """
        kwargs = dict(apply_mask=False, use_fp16=self.use_fp16)
        if self.name.startswith("moge3"):
            kwargs["refine_steps"] = self.refine_steps
        prediction = self.model.infer(image.to(self.device), **kwargs)
        depth = prediction["depth"].float().detach().cpu()
        normalized_k = prediction["intrinsics"].float().detach().cpu()
        mask = prediction.get("mask")
        mask = torch.ones_like(depth, dtype=torch.bool) if mask is None else mask.detach().cpu().bool()
        height, width = depth.shape
        if normalized_k.shape != (3, 3):
            raise ValueError(f"MoGe must return one [3,3] K, got {tuple(normalized_k.shape)}.")
        intrinsic = normalized_k.clone()
        intrinsic[0] *= width
        intrinsic[1] *= height
        if not torch.isfinite(intrinsic).all() or (intrinsic[:2, :2].diagonal() <= 0).any():
            raise ValueError("MoGe returned invalid camera intrinsics.")
        # Use the model's camera-z depth, not Euclidean point-map distance. Its
        # K and z jointly define a pinhole point map consistent with Uni3C.
        depth, valid, filled = repair_depth(depth, mask)
        return DepthPrediction(depth, intrinsic, valid, self.name, filled)


def repair_depth(depth, valid_mask=None):
    """Replace invalid/nonpositive predictions deterministically for rendering.

    Uni3C's point renderer cannot consume inf/NaN. Invalid pixels (typically sky)
    are assigned the conservative far cap so they stay behind valid geometry.
    The original validity mask is returned and recorded; no metric accuracy is
    claimed for filled regions.
    """
    depth = depth.float().detach().cpu()
    if depth.ndim != 2:
        raise ValueError(f"Depth must be [H,W], got {tuple(depth.shape)}.")
    valid = torch.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid &= valid_mask.bool()
    if not valid.any():
        raise ValueError("Depth model returned no finite positive pixels.")
    # Keep filled sky/invalid pixels behind every retained finite surface.
    far = depth[valid].max().clamp_min(1e-3)
    filled = int((~valid).sum())
    depth = torch.where(valid, depth, far).clamp_min(1e-4)
    return depth, valid, filled
