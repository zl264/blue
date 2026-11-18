"""Portable MoGe-3 dense inference without CUDA sparse-refiner imports.

The official ``moge.model.v3`` imports FlexGEMM/Triton at module import time,
even when ``refine_steps=0``. The scoped import below substitutes only that
unused module. Dense encoder/neck/heads remain official MoGe code. Focal/shift
recovery uses an on-device damped Gauss-Newton solver for the official objective
instead of the upstream NumPy/SciPy postprocessor.
"""
import importlib
import os
import sys
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


def _normalized_view_plane_uv(height, width, device):
    """Match MoGe's normalized UV grid without constructing host arrays."""
    aspect_ratio = width / height
    diagonal_scale = (1 + aspect_ratio ** 2) ** 0.5
    span_x, span_y = aspect_ratio / diagonal_scale, 1 / diagonal_scale
    u = torch.linspace(
        -span_x * (width - 1) / width, span_x * (width - 1) / width,
        width, dtype=torch.float32, device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height,
        height, dtype=torch.float32, device=device,
    )
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1)


def recover_focal_shift_npu(points, mask=None, downsample_size=(64, 64),
                            max_iterations=100):
    """Solve MoGe's focal/shift objective entirely on the input device.

    The official helper downsamples in Torch, then copies those maps to NumPy
    and runs SciPy's scalar Levenberg-Marquardt solver. This implementation
    keeps the same residuals and nearest-neighbor downsampling, but performs
    the one-variable damped Gauss-Newton/LM iterations in Torch. For each shift,
    focal has the same closed-form optimum as the official objective.
    """
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[0] != 1:
        raise ValueError("Portable focal recovery expects points [1,H,W,3].")
    points = points.float()
    height, width = points.shape[-3:-1]
    uv = _normalized_view_plane_uv(height, width, points.device)
    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest",
    ).permute(0, 2, 3, 1)[0]
    uv_lr = F.interpolate(
        uv[None].permute(0, 3, 1, 2), downsample_size, mode="nearest",
    ).permute(0, 2, 3, 1)[0]
    if mask is not None:
        mask_lr = F.interpolate(
            mask.float()[None, None], downsample_size, mode="nearest",
        )[0, 0] > 0
        points_lr, uv_lr = points_lr[mask_lr], uv_lr[mask_lr]
    else:
        points_lr, uv_lr = points_lr.reshape(-1, 3), uv_lr.reshape(-1, 2)
    if points_lr.shape[0] < 2:
        one = points.new_ones(())
        return one, points.new_zeros(())

    xy, z = points_lr[..., :2], points_lr[..., 2]
    uv = uv_lr.reshape(-1, 2)

    def evaluate(shift):
        denominator = z + shift
        # Avoid a host fallback or an unreported NaN at the singular shift.
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.where(denominator < 0, -1e-6, 1e-6),
            denominator,
        )
        projection = xy / safe[:, None]
        numerator = (projection * uv).sum()
        denominator_focal = projection.square().sum()
        focal = numerator / denominator_focal.clamp_min(1e-12)
        residual = focal * projection - uv
        cost = 0.5 * residual.square().sum()
        return projection, focal, residual, cost, denominator_focal, safe

    shift = points.new_zeros(())
    damping = 1e-3
    converged = False
    for _ in range(max_iterations):
        projection, focal, residual, cost, norm2, safe = evaluate(shift)
        if not bool(torch.isfinite(cost)):
            raise RuntimeError("MoGe focal/shift optimization produced non-finite cost.")
        projection_derivative = -projection / safe[:, None]
        numerator_derivative = (projection_derivative * uv).sum()
        norm2_derivative = 2 * (projection * projection_derivative).sum()
        numerator = (projection * uv).sum()
        focal_derivative = (
            numerator_derivative * norm2 - numerator * norm2_derivative
        ) / norm2.square().clamp_min(1e-12)
        derivative = focal_derivative * projection + focal * projection_derivative
        gradient = (derivative * residual).sum()
        curvature = derivative.square().sum().clamp_min(1e-12)
        step = -gradient / (curvature * (1 + damping))
        candidate_shift = shift + step
        *_, candidate_cost, _, _ = evaluate(candidate_shift)
        if bool(torch.isfinite(candidate_cost)) and bool(candidate_cost <= cost):
            reduction = cost - candidate_cost
            shift = candidate_shift
            damping = max(damping * 0.3, 1e-12)
            # Match the official ftol=1e-3 convergence criterion in spirit;
            # a scalar device sync is not a CPU geometry/tensor fallback.
            if bool(reduction <= 1e-3 * candidate_cost.clamp_min(1e-12)):
                converged = True
                break
        else:
            damping = min(damping * 10, 1e12)
    if not converged:
        raise RuntimeError("MoGe focal/shift optimization did not converge.")
    _, focal, _, _, _, _ = evaluate(shift)
    if not bool(torch.isfinite(focal)) or not bool(torch.isfinite(shift)):
        raise RuntimeError("MoGe focal/shift optimization returned invalid camera values.")
    return focal, shift


def _disabled_sparse_refiner(*args, **kwargs):
    raise RuntimeError(
        "MoGe-3 sparse refinement is unavailable in the portable dense loader. "
        "Use the official CUDA/FlexGEMM installation for refinement."
    )


def load_moge3_dense(checkpoint, device, refine_steps=0):
    """Load every dense checkpoint tensor strictly on an NPU or CPU.

    Only ``refiner.*`` is excluded. The stub sparse module never enters the
    dense forward and is restored/removed from ``sys.modules`` after loading.
    """
    if refine_steps != 0:
        raise ValueError("Portable MoGe-3 inference requires refine_steps=0.")
    if torch.device(device).type not in ("npu", "cpu"):
        raise ValueError("Use the official MoGe loader for non-portable devices.")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        checkpoint_path = Path(hf_hub_download(
            repo_id=str(checkpoint), repo_type="model", filename="model.pt",
        ))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("model_config"), dict):
        raise ValueError("MoGe-3 checkpoint must contain a model_config dictionary.")
    if not isinstance(state.get("model"), dict):
        raise ValueError("MoGe-3 checkpoint must contain a model state dictionary.")

    sparse_stub = types.ModuleType("moge.model.modules.sparse_unet")
    sparse_stub.Sparse3DUNet = _disabled_sparse_refiner
    previous_xformers = os.environ.get("XFORMERS_DISABLED")
    try:
        # ViT-g probes xFormers while importing DINOv2. Its pure PyTorch
        # SwiGLU is equivalent and is selected on non-CUDA devices.
        os.environ["XFORMERS_DISABLED"] = "1"
        with mock.patch.dict(sys.modules, {
            "moge.model.modules.sparse_unet": sparse_stub,
        }):
            moge_v3 = importlib.import_module("moge.model.v3")
            # Official MoGeModel only creates self.refiner when this config is
            # non-null. Disable construction itself; strict dense loading below
            # then verifies that no non-refiner checkpoint tensors were dropped.
            config = dict(state["model_config"])
            config["refiner"] = None
            model = moge_v3.MoGeModel(**config)
    except ImportError as error:
        raise ImportError(
            "MoGe-3 or its portable geometry dependencies are missing. Follow "
            "the --no-deps installation in docs/ascend_static_scene.md; do not "
            "install flex-gemm, Triton, xFormers, or CUDA PyTorch wheels on NPU."
        ) from error
    finally:
        if previous_xformers is None:
            os.environ.pop("XFORMERS_DISABLED", None)
        else:
            os.environ["XFORMERS_DISABLED"] = previous_xformers

    if hasattr(model, "refiner"):
        raise RuntimeError("Portable MoGe-3 unexpectedly constructed a sparse refiner.")
    weights = {
        key: value for key, value in state["model"].items()
        if not key.startswith("refiner.")
    }
    model.load_state_dict(weights, strict=True)
    excluded = len(state["model"]) - len(weights)
    print(
        f"MoGe-3 portable dense inference: all {len(weights)} dense tensors "
        f"loaded strictly; excluded {excluded} unused sparse-refiner tensors."
    )
    return model.to(device=device, dtype=torch.float32).eval()


def infer_moge3_dense(model, image):
    """Return camera-z, normalized K and mask using the dense model on-device.

    This follows MoGe-3 ``infer(refine_steps=0, apply_mask=False)`` for the
    fields consumed by Uni3C, without an FP32 NPU autocast context. No points,
    normals or depth values are copied to the host during inference.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("MoGe-3 portable inference expects one RGB [3,H,W] image.")
    device = next(model.parameters()).device
    image = image.to(device=device, dtype=next(model.parameters()).dtype)
    height, width = image.shape[-2:]
    min_tokens, max_tokens = model.num_tokens_range
    num_tokens = max_tokens  # Official resolution_level=9.
    output = model.forward(image, num_tokens=num_tokens, refine_steps=0)
    points = output["points"].float()
    mask = output.get("mask")
    mask = mask.float() > 0.5 if mask is not None else None

    focal, shift = recover_focal_shift_npu(points, mask)
    aspect_ratio = width / height
    diagonal_scale = (1 + aspect_ratio ** 2) ** 0.5
    fx = focal / 2 * diagonal_scale / aspect_ratio
    fy = focal / 2 * diagonal_scale
    intrinsics = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0
    depth = points[..., 2] + shift[..., None, None]
    if mask is not None:
        mask = mask & (depth > 0)
    metric_scale = output.get("metric_scale")
    if metric_scale is not None:
        depth = depth * metric_scale.float()[:, None, None]
    if not bool(torch.isfinite(depth).any()):
        raise ValueError("MoGe-3 dense inference returned no finite depth.")
    return {
        "depth": depth[0],
        "intrinsics": intrinsics[0],
        "mask": mask[0] if mask is not None else None,
    }
