import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..core.data.operators import DataProcessingOperator


@dataclass
class UCPECameraTrajectory:
    """A camera trajectory before crop/resize and latent-grid alignment.

    Poses are camera-to-world matrices in the OpenCV convention (x right,
    y down, z forward). Intrinsics are pixel-space ``(fx, fy, cx, cy)``.
    ``x_fov`` can be used instead for UCPE's original UCM convention.
    """

    poses: torch.Tensor
    intrinsics: Optional[torch.Tensor]
    source_size: Optional[Tuple[int, int]]
    frame_indices: torch.Tensor
    x_fov: Optional[torch.Tensor] = None
    xi: Optional[torch.Tensor] = None
    intrinsics_already_cropped: bool = False
    intrinsics_normalized: bool = False
    translation_scale: float = 1.0


class LoadUCPECameraTrajectory(DataProcessingOperator):
    """Load JSON/JSONL/NPZ or RealEstate10K-style camera trajectories.

    JSON accepts a top-level ``c2w``/``w2c`` array or a ``frames`` list. Each
    frame can contain ``c2w``/``w2c``, ``intrinsics`` and ``frame_index``.
    Intrinsics can be four values ``(fx, fy, cx, cy)`` or a 3x3 K matrix.
    Per-frame 3x3 K matrices are supported; use ``[3,3]`` for a shared K.
    Set ``intrinsics_normalized`` when they are divided by image width/height.
    ``image_size: [width, height]`` describes the uncropped input video.
    """

    def __call__(self, path: str):
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                metadata = {key: data[key] for key in data.files}
            metadata = {
                key: (value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value)
                for key, value in metadata.items()
            }
        elif suffix in (".json", ".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                if suffix == ".jsonl":
                    metadata = {"frames": [json.loads(line) for line in f if line.strip()]}
                else:
                    metadata = json.load(f)
        elif suffix == ".txt":
            return self._load_re10k(path)
        else:
            raise ValueError(f"Unsupported camera trajectory file: {path}. Use .json, .jsonl, .npz or RealEstate10K .txt.")
        return self.from_dict(metadata, source=path)

    def _load_re10k(self, path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith(("http://", "https://")):
                    continue
                values = line.split()
                if len(values) < 19:
                    raise ValueError(f"Invalid RealEstate10K camera row in {path}: expected at least 19 values, got {len(values)}.")
                rows.append([float(value) for value in values[:19]])
        if len(rows) == 0:
            raise ValueError(f"No camera poses found in {path}.")
        rows = np.asarray(rows, dtype=np.float64)
        w2c = np.concatenate(
            [rows[:, 7:].reshape(-1, 3, 4), np.tile([0, 0, 0, 1], (len(rows), 1, 1))], axis=1,
        )
        # RealEstate10K stores normalized intrinsics and world-to-camera poses.
        metadata = {
            "w2c": w2c,
            "intrinsics": rows[:, 1:5],
            "intrinsics_normalized": True,
            # The video loader takes the RGB frames in file order without FPS
            # resampling, so match row order rather than treating RE10K's
            # microsecond timestamps as RGB frame indices.
            "frame_indices": np.arange(len(rows), dtype=np.int64),
        }
        return self.from_dict(metadata, source=path)

    @classmethod
    def from_dict(cls, metadata, source="camera metadata"):
        if isinstance(metadata, list):
            metadata = {"frames": metadata}
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}: expected a JSON object or frame list.")
        metadata = dict(metadata)
        frames = metadata.get("frames")
        if frames is not None:
            if len(frames) == 0:
                raise ValueError(f"{source}: the camera frame list is empty.")
            for key in ("c2w", "w2c", "intrinsics", "frame_indices"):
                if key in metadata:
                    raise ValueError(f"{source}: do not mix top-level {key} with a frames list.")
            pose_key = "c2w" if "c2w" in frames[0] else "w2c" if "w2c" in frames[0] else None
            if pose_key is None:
                raise ValueError(f"{source}: every frame must contain c2w or w2c.")
            metadata[pose_key] = [frame[pose_key] for frame in frames]
            if "intrinsics" in frames[0]:
                metadata["intrinsics"] = [frame["intrinsics"] for frame in frames]
            metadata["frame_indices"] = [frame.get("frame_index", frame.get("frame_idx", frame.get("frame_id", i))) for i, frame in enumerate(frames)]
            for key in ("x_fov", "xi"):
                if key in frames[0]:
                    metadata[key] = [frame[key] for frame in frames]
        pose_key = "c2w" if "c2w" in metadata else "w2c" if "w2c" in metadata else None
        if pose_key is None:
            raise ValueError(f"{source}: missing c2w or w2c poses.")
        poses = torch.as_tensor(np.asarray(metadata[pose_key], dtype=np.float64)).float()
        if poses.ndim == 3 and poses.shape[1:] == (3, 4):
            poses = torch.cat([poses, torch.tensor([0, 0, 0, 1]).view(1, 1, 4).expand(len(poses), 1, 4)], dim=1)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
            raise ValueError(f"{source}: poses must have shape [T,4,4] or [T,3,4], got {tuple(poses.shape)}.")
        if pose_key == "w2c":
            poses = torch.linalg.inv(poses)
        convention = metadata.get("coordinate_system", metadata.get("convention", "opencv")).lower()
        aliases = {"cv": "opencv", "colmap": "opencv", "opengl": "opengl", "nerf": "opengl"}
        convention = aliases.get(convention, convention)
        if convention == "opengl":
            conversion = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
            poses = poses @ conversion
        elif convention != "opencv":
            raise ValueError(f"{source}: unsupported coordinate_system={convention!r}; use opencv or opengl.")
        if not torch.isfinite(poses).all():
            raise ValueError(f"{source}: poses contain NaN or infinity.")

        num_frames = len(poses)
        intrinsics = metadata.get("intrinsics", metadata.get("K"))
        intrinsics_normalized = bool(metadata.get("intrinsics_normalized", False))
        if intrinsics is not None:
            intrinsics = torch.as_tensor(np.asarray(intrinsics, dtype=np.float64)).float()
            if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
                intrinsics = torch.stack([
                    intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2],
                ], dim=-1)
            elif intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
                intrinsics = intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]].unsqueeze(0)
            if intrinsics.ndim == 1:
                intrinsics = intrinsics.unsqueeze(0)
            if intrinsics.ndim != 2 or intrinsics.shape[-1] != 4 or len(intrinsics) not in (1, num_frames):
                raise ValueError(f"{source}: intrinsics must be [4], [T,4], [3,3], or [T,3,3], got {tuple(intrinsics.shape)}.")
            intrinsics = intrinsics.expand(num_frames, 4).clone()
            # Keep normalized K normalized until the training resolution is
            # known. RealEstate10K files do not contain the source resolution.
        x_fov = metadata.get("x_fov")
        x_fov = None if x_fov is None else torch.as_tensor(x_fov, dtype=torch.float32).reshape(-1)
        if x_fov is not None:
            if len(x_fov) not in (1, num_frames):
                raise ValueError(f"{source}: x_fov must be scalar or have one value per frame.")
            x_fov = x_fov.expand(num_frames).clone()
        if intrinsics is None and x_fov is None:
            raise ValueError(f"{source}: provide intrinsics/K or UCPE-style x_fov.")
        xi = torch.as_tensor(metadata.get("xi", 0.0), dtype=torch.float32).reshape(-1)
        if len(xi) not in (1, num_frames):
            raise ValueError(f"{source}: xi must be scalar or have one value per frame.")
        xi = xi.expand(num_frames).clone()
        if intrinsics is not None and (intrinsics[:, :2] <= 0).any():
            raise ValueError(f"{source}: fx and fy must be positive.")
        if x_fov is not None and ((x_fov <= 0) | (x_fov >= 180)).any():
            raise ValueError(f"{source}: x_fov must be between 0 and 180 degrees.")

        image_size = metadata.get("image_size")
        if image_size is None and metadata.get("width") is not None and metadata.get("height") is not None:
            image_size = [metadata["width"], metadata["height"]]
        source_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        if source_size is not None and min(source_size) <= 0:
            raise ValueError(f"{source}: image_size must be positive.")
        already_cropped = metadata.get("intrinsics_already_cropped", metadata.get("intrinsics_mode", "original") == "target")
        if not already_cropped and intrinsics is not None and source_size is None and not intrinsics_normalized:
            raise ValueError(f"{source}: image_size: [width, height] is required for pixel-space intrinsics. Set intrinsics_already_cropped=true only if K already matches the training resolution.")
        frame_indices = torch.as_tensor(metadata.get("frame_indices", np.arange(num_frames)), dtype=torch.long).reshape(-1)
        if len(frame_indices) != num_frames or len(torch.unique(frame_indices)) != num_frames:
            raise ValueError(f"{source}: frame_indices must be unique and have one entry per pose.")
        translation_scale = float(metadata.get("translation_scale", 1.0))
        if translation_scale <= 0:
            raise ValueError(f"{source}: translation_scale must be positive.")
        return cls(
            poses=poses,
            intrinsics=intrinsics,
            source_size=source_size,
            frame_indices=frame_indices,
            x_fov=x_fov,
            xi=xi,
            intrinsics_already_cropped=bool(already_cropped),
            intrinsics_normalized=intrinsics_normalized,
            translation_scale=translation_scale,
        )


def _crop_resize_intrinsics(intrinsics, source_size, height, width):
    """Match ImageCropAndResize (resize-to-cover followed by center crop)."""
    intrinsics = intrinsics.clone()
    if source_size is None:
        return intrinsics
    source_width, source_height = source_size
    scale = max(width / source_width, height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    crop_left = int(round((resized_width - width) / 2.0))
    crop_top = int(round((resized_height - height) / 2.0))
    intrinsics = intrinsics * torch.tensor([scale, scale, scale, scale], device=intrinsics.device)
    intrinsics[:, 2] -= crop_left
    intrinsics[:, 3] -= crop_top
    return intrinsics


def _ucm_unproject(intrinsics, xi, height, width):
    """UCPE/equilib unified-camera unprojection on an integer pixel grid."""
    num_frames = len(intrinsics)
    y, x = torch.meshgrid(
        torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype),
        torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype), indexing="ij",
    )
    x = x.unsqueeze(0).expand(num_frames, -1, -1)
    y = y.unsqueeze(0).expand(num_frames, -1, -1)
    fx, fy, cx, cy = (intrinsics[:, i, None, None] for i in range(4))
    xi = xi[:, None, None]
    x, y = (x - cx) / fx, (y - cy) / fy
    radius2 = x.square() + y.square()
    radicand = 1 + (1 - xi.square()) * radius2
    if (radicand < 0).any():
        raise ValueError("Invalid UCM intrinsics: the unprojection square root is negative. Check xi and field of view.")
    alpha = xi + torch.sqrt(radicand)
    gamma = alpha / (1 + radius2)
    return torch.stack([gamma * x, gamma * y, gamma - xi], dim=-1)


def _world_to_ray_matrices(directions, c2w):
    """UCPE Relative Ray Encoding: one world-to-ray SE(3) matrix per patch."""
    rotation = c2w[:, :, :3, :3]
    center = c2w[:, :, :3, 3]
    directions_world = torch.einsum("btij,bthwj->bthwi", rotation, directions)
    camera_y = rotation[:, :, :, 1][:, :, None, None, :].expand_as(directions_world)
    z_axis = F.normalize(directions_world, dim=-1, eps=1e-6)
    x_axis = F.normalize(torch.cross(camera_y, z_axis, dim=-1), dim=-1, eps=1e-6)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-6)
    local_to_world = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    world_to_local = local_to_world.transpose(-1, -2)
    translation = -torch.einsum("bthwij,bthwj->bthwi", world_to_local, center[:, :, None, None, :].expand_as(directions_world))
    matrices = torch.zeros(*directions.shape[:-1], 4, 4, device=c2w.device, dtype=c2w.dtype)
    matrices[..., :3, :3] = world_to_local
    matrices[..., :3, 3] = translation
    matrices[..., 3, 3] = 1
    return matrices, directions_world


def _absolute_orientation_map(directions_world, rotation, intrinsics, xi):
    """UCPE's three-channel world-up/absolute-latitude map."""
    dtype = directions_world.dtype
    directions_world = F.normalize(directions_world, dim=-1, eps=1e-8)
    x, y, z = directions_world.unbind(dim=-1)
    latitude = torch.atan2(-y, torch.sqrt(x.square() + z.square())).unsqueeze(-1)
    up_world = torch.tensor([0, -1, 0], device=directions_world.device, dtype=dtype)
    tangent = torch.cross(directions_world, up_world.expand_as(directions_world), dim=-1)
    tangent = F.normalize(tangent, dim=-1, eps=1e-8)
    delta = 0.1
    rotated = (
        directions_world * math.cos(delta)
        + torch.cross(tangent, directions_world, dim=-1) * math.sin(delta)
        + tangent * (tangent * directions_world).sum(dim=-1, keepdim=True) * (1 - math.cos(delta))
    )
    directions_camera = torch.einsum("btji,bthwj->bthwi", rotation, rotated)
    x, y, z = directions_camera.unbind(dim=-1)
    radius = torch.sqrt(x.square() + y.square() + z.square())
    denominator = z + xi[:, :, None, None] * radius
    fx, fy, cx, cy = (intrinsics[..., i][:, :, None, None] for i in range(4))
    u = fx * x / denominator + cx
    v = fy * y / denominator + cy
    grid_y, grid_x = torch.meshgrid(
        torch.arange(u.shape[-2], device=u.device, dtype=dtype),
        torch.arange(u.shape[-1], device=u.device, dtype=dtype), indexing="ij",
    )
    up = torch.stack([u - grid_x, v - grid_y], dim=-1)
    up = F.normalize(up, dim=-1, eps=1e-8)
    return torch.cat([up, latitude], dim=-1)


def build_ucpe_ray_features(
    trajectory: UCPECameraTrajectory,
    num_frames: int,
    height: int,
    width: int,
    latent_frames: int,
    vae_stride: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    condition="relray_absmap",
    normalize_to_first=True,
    translation_scale=1.0,
    device="cpu",
):
    """Return float32 ``[1,T,H_patch,W_patch,C]`` UCPE camera features."""
    if condition not in ("relray", "relray_absmap"):
        raise ValueError(f"Unknown UCPE camera condition: {condition}")
    if translation_scale <= 0:
        raise ValueError("translation_scale must be positive.")
    if latent_frames != (num_frames - 1) // vae_stride[0] + 1 or num_frames % vae_stride[0] != 1:
        raise ValueError(f"num_frames={num_frames} and latent_frames={latent_frames} do not match VAE stride {vae_stride[0]}.")
    frame_ids = torch.arange(latent_frames, dtype=torch.long) * vae_stride[0]
    available = {int(frame): index for index, frame in enumerate(trajectory.frame_indices.tolist())}
    missing = [int(frame) for frame in frame_ids if int(frame) not in available]
    if missing:
        raise ValueError(f"Camera trajectory is missing sampled RGB frames {missing[:8]}. Required frames are 0, {vae_stride[0]}, ..., {num_frames - 1}; got {len(trajectory.poses)} poses.")
    pose_indices = torch.tensor([available[int(frame)] for frame in frame_ids], dtype=torch.long)
    poses = trajectory.poses[pose_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
    if normalize_to_first:
        poses = torch.linalg.inv(poses[:, :1]) @ poses
    poses = poses.clone()
    poses[..., :3, 3] *= translation_scale * trajectory.translation_scale
    patch_height = height // (vae_stride[1] * patch_size[1])
    patch_width = width // (vae_stride[2] * patch_size[2])
    if height % (vae_stride[1] * patch_size[1]) != 0 or width % (vae_stride[2] * patch_size[2]) != 0:
        raise ValueError(
            f"Video size {(height, width)} must be divisible by the VAE x DiT spatial footprint "
            f"{(vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])}."
        )
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"Invalid ray grid {(patch_height, patch_width)} for video size {(height, width)}.")
    if trajectory.x_fov is not None:
        if (
            trajectory.source_size is not None and not trajectory.intrinsics_already_cropped
            and not math.isclose(
                trajectory.source_size[0] / trajectory.source_size[1], width / height, rel_tol=1e-6,
            )
        ):
            raise ValueError(
                "x_fov + xi cannot be center-cropped without an aspect-ratio change. "
                "Convert UCM parameters to pixel intrinsics and provide intrinsics/image_size instead."
            )
        x_fov = trajectory.x_fov[pose_indices].to(device=device, dtype=torch.float32)
        xi = trajectory.xi[pose_indices].to(device=device, dtype=torch.float32)
        theta = torch.deg2rad(0.5 * x_fov)
        fx = (patch_width * 0.5) * (torch.cos(theta) + xi) / torch.sin(theta).clamp_min(torch.finfo(torch.float32).eps)
        intrinsics = torch.stack([
            fx, fx,
            torch.full_like(fx, patch_width / 2),
            torch.full_like(fx, patch_height / 2),
        ], dim=-1)
    else:
        intrinsics = trajectory.intrinsics[pose_indices].to(device=device, dtype=torch.float32).clone()
        if trajectory.intrinsics_normalized:
            # Without source_size, normalized K defines the same aspect ratio
            # as the target video; no absolute source resolution is needed.
            source_width, source_height = (
                (width, height) if trajectory.intrinsics_already_cropped
                else trajectory.source_size or (width, height)
            )
            intrinsics *= torch.tensor(
                [source_width, source_height, source_width, source_height], device=device,
            )
        if trajectory.source_size is not None and not trajectory.intrinsics_already_cropped:
            intrinsics = _crop_resize_intrinsics(intrinsics, trajectory.source_size, height, width)
        intrinsics = intrinsics / torch.tensor([
            vae_stride[2] * patch_size[2], vae_stride[1] * patch_size[1],
            vae_stride[2] * patch_size[2], vae_stride[1] * patch_size[1],
        ], device=device)
        xi = trajectory.xi[pose_indices].to(device=device, dtype=torch.float32)
    # A batch dimension is kept even though current Wan training uses batch size one.
    intrinsics = intrinsics.unsqueeze(0)
    xi = xi.unsqueeze(0)
    directions = _ucm_unproject(intrinsics.reshape(-1, 4), xi.reshape(-1), patch_height, patch_width)
    directions = directions.reshape(1, latent_frames, patch_height, patch_width, 3)
    ray_matrices, directions_world = _world_to_ray_matrices(directions, poses)
    features = ray_matrices.flatten(-2)
    if condition == "relray_absmap":
        absolute_map = _absolute_orientation_map(directions_world, poses[:, :, :3, :3], intrinsics, xi)
        features = torch.cat([features, absolute_map], dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("UCPE ray features contain NaN or infinity. Check camera rotations, UCM xi/FOV and degenerate vertical rays.")
    return features


UCPE_LORA_TARGETS = (
    ("self_attn", ("q", "k", "v", "o")),
    ("ffn", ("0", "2")),
)


def validate_ucpe_lora_coverage(dit):
    """Verify all six PEFT adapters on every Wan2.2-5B DiT block before fusion."""
    if dit.dim != 3072 or len(dit.blocks) != 30:
        raise ValueError("UCPE requires the Wan2.2 TI2V-5B DiT with dim=3072 and exactly 30 layers.")
    missing = []
    for block_id, block in enumerate(dit.blocks):
        for parent_name, names in UCPE_LORA_TARGETS:
            parent = getattr(block, parent_name)
            for name in names:
                module = getattr(parent, name)
                if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                    missing.append(f"blocks.{block_id}.{parent_name}.{name}")
    if missing:
        raise ValueError(
            "UCPE requires LoRA on q,k,v,o,ffn.0,ffn.2 in every one of the 30 DiT layers; "
            f"missing: {missing[:8]}. Use Wan2.2-TI2V-5B-UCPE.sh."
        )


def validate_ucpe_lora_checkpoint(state_dict, prefix=""):
    """Validate saved DiT LoRA coverage before DiffSynth fuses it for inference."""
    expected = {
        f"{prefix}blocks.{block_id}.{parent_name}.{name}.lora_{side}.weight"
        for block_id in range(30)
        for parent_name, names in UCPE_LORA_TARGETS
        for name in names
        for side in ("A", "B")
    }
    # PEFT's native saved keys may include the explicit ``default`` adapter.
    missing = [
        name for name in sorted(expected)
        if name not in state_dict and name.replace(".weight", ".default.weight") not in state_dict
    ]
    if missing:
        raise ValueError(
            f"UCPE checkpoint must contain all six LoRA target modules across all 30 DiT layers; "
            f"missing {len(missing)} of {len(expected)} tensors, e.g. {missing[:4]}."
        )


class UCPECameraEncoder(nn.Module):
    """Project UCPE ray features once, then add them to existing DiT patches."""

    def __init__(self, in_dim, dim, hidden_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(in_dim)
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features):
        return self.output_projection(self.activation(self.input_projection(self.norm(features))))
