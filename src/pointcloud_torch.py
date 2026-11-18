"""Pure PyTorch point splatting for devices without a PyTorch3D backend.

This is a geometry-compatible fallback, not a bit-exact replacement for
PyTorch3D's PointsRenderer. It uses nearest-depth visibility and weighted point
splatting with bounded memory, and runs entirely on the input tensor device.
"""
import torch


def _visible_candidates(points, colors, w2c, intrinsic, height, width,
                        offsets):
    """Project one point chunk and return flattened, in-frame splat candidates."""
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    camera_points = points @ rotation.transpose(0, 1) + translation
    z = camera_points[:, 2]
    safe_z = z.clamp_min(1e-8)
    center_u = intrinsic[0, 0] * camera_points[:, 0] / safe_z + intrinsic[0, 2] - 0.5
    center_v = intrinsic[1, 1] * camera_points[:, 1] / safe_z + intrinsic[1, 2] - 0.5
    centers = torch.stack([center_u, center_v], dim=-1).round()
    candidates = centers[:, None, :] + offsets[None, :, :]
    x = candidates[..., 0]
    y = candidates[..., 1]
    candidate_z = z[:, None].expand_as(x)
    valid = ((z[:, None] > 1e-6) & torch.isfinite(candidates).all(dim=-1)
             & (x >= 0) & (x < width) & (y >= 0) & (y < height))
    x = x[valid].long()
    y = y[valid].long()
    z = candidate_z[valid]
    pixels = y * width + x
    expanded_colors = colors[:, None, :].expand(-1, offsets.shape[0], -1)
    candidate_colors = expanded_colors[valid]
    distances2 = (candidates - centers[:, None, :]).square().sum(dim=-1)[valid]
    return pixels, z, candidate_colors, distances2


def _render_frame(points, colors, intrinsic, w2c, height, width,
                  background_color, splat_radius=1, chunk_size=32768):
    """Render one camera using a two-pass nearest-z, weighted-splat z buffer."""
    device = points.device
    pixel_count = height * width
    z_buffer = torch.full((pixel_count,), torch.inf, dtype=torch.float32, device=device)
    if splat_radius < 0 or int(splat_radius) != splat_radius:
        raise ValueError("splat_radius must be a nonnegative integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    splat_radius = int(splat_radius)
    axes = torch.arange(-splat_radius, splat_radius + 1, device=device, dtype=torch.float32)
    offsets = torch.stack(torch.meshgrid(axes, axes, indexing="ij"), dim=-1).reshape(-1, 2)

    # Pass 1: nearest camera-z per output pixel. scatter_reduce avoids allocating
    # an [H,W,points_per_pixel] tensor and keeps peak memory bounded by chunk_size.
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        pixels, z, _, _ = _visible_candidates(
            points[start:end], colors[start:end], w2c, intrinsic, height, width,
            offsets,
        )
        z_buffer.scatter_reduce_(0, pixels, z, reduce="amin", include_self=True)

    color_sum = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((pixel_count,), dtype=torch.float32, device=device)
    footprint = float(splat_radius) + 0.5
    # Pass 2: composite only splats on the nearest visible surface. A relative
    # tolerance joins duplicate samples of that surface without blending through
    # a genuine depth discontinuity.
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        pixels, z, candidate_colors, distances2 = _visible_candidates(
            points[start:end], colors[start:end], w2c, intrinsic, height, width,
            offsets,
        )
        nearest_z = z_buffer[pixels]
        tolerance = nearest_z.abs().clamp_min(1.0) * 1e-4
        visible = z <= nearest_z + tolerance
        if not bool(visible.any()):
            continue
        pixels = pixels[visible]
        candidate_colors = candidate_colors[visible]
        if splat_radius == 0:
            weights = torch.ones_like(pixels, dtype=torch.float32)
        else:
            weights = (1.0 - distances2[visible] / (footprint * footprint)).clamp_min(0.0)
        weight_sum.index_add_(0, pixels, weights)
        color_sum.index_add_(0, pixels, candidate_colors * weights[:, None])

    occupied = weight_sum > 0
    background = torch.as_tensor(background_color, dtype=torch.float32, device=device)
    image = background[None, :].expand(pixel_count, 3).clone()
    image[occupied] = color_sum[occupied] / weight_sum[occupied, None]
    image = image.reshape(height, width, 3).permute(2, 0, 1).contiguous()
    mask = (~occupied).float().reshape(1, height, width)
    return image, mask


def render_points_torch(points, colors, intrinsic, w2cs, height, width,
                        background_color=(0, 0, 0), splat_radius=1,
                        chunk_size=32768, progress=False):
    """Render world-space points under OpenCV w2c cameras on their current device.

    Args:
        points: [P,3] finite world points, in the same units as camera translation.
        colors: [P,3] RGB values in the pipeline's [-1,1] range.
        intrinsic: [N,3,3] pixel-unit pinhole intrinsics.
        w2cs: [N,4,4] OpenCV world-to-camera matrices.
        splat_radius: integer output-pixel footprint radius; 0 is nearest-point.
        chunk_size: source points per pass, controlling temporary candidate memory.
    Returns:
        RGB [N,3,H,W] and missing-geometry masks [N,1,H,W] (1 means missing).
    """
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("points and colors must have matching [P,3] shapes")
    if intrinsic.shape != (len(w2cs), 3, 3) or w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4):
        raise ValueError("Expected intrinsic [N,3,3] and w2cs [N,4,4]")
    if not torch.isfinite(intrinsic).all() or not torch.isfinite(w2cs).all():
        raise ValueError("Camera calibration must be finite")
    if not torch.isfinite(colors).all():
        raise ValueError("Point colors must be finite")
    if height <= 0 or width <= 0 or not torch.isfinite(points).all():
        raise ValueError("Image dimensions must be positive and points must be finite")
    points = points.float()
    colors = colors.float()
    intrinsic = intrinsic.float()
    w2cs = w2cs.float()
    images, masks = [], []
    for index, w2c in enumerate(w2cs):
        image, mask = _render_frame(
            points, colors, intrinsic[index], w2c, height, width,
            background_color, splat_radius=splat_radius, chunk_size=chunk_size,
        )
        images.append(image)
        masks.append(mask)
        if progress and (index == 0 or (index + 1) % 10 == 0 or index + 1 == len(w2cs)):
            print(f"Pure PyTorch point rendering: {index + 1}/{len(w2cs)} frames", flush=True)
    return torch.stack(images), torch.stack(masks)
