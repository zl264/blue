"""Torch subject lifting/look-at geometry for NPU static preprocessing."""
import torch


def resize_subject_pixel(pixel, source_size, target_size, device):
    source_w, source_h = source_size
    target_w, target_h = target_size
    uv = torch.as_tensor(pixel, dtype=torch.float32, device=device)
    scale = torch.tensor(
        [target_w / source_w, target_h / source_h],
        dtype=torch.float32, device=device,
    )
    return (uv + 0.5) * scale - 0.5


def lift_subject_to_world(pixel, depth, intrinsic, c2w, source_size,
                          depth_override=None, sample_radius=2):
    """Lift one clicked pixel with the same convention as the NumPy helper."""
    if depth.ndim == 4 and depth.shape[:2] == (1, 1):
        depth = depth[0, 0]
    if depth.ndim != 2 or not isinstance(sample_radius, int) or sample_radius < 0:
        raise ValueError("Depth must be [H,W] and sample_radius nonnegative.")
    source_w, source_h = source_size
    if min(source_w, source_h) <= 0:
        raise ValueError("Source image dimensions must be positive.")
    height, width = depth.shape
    pixel = torch.as_tensor(pixel, dtype=torch.float32, device=depth.device)
    bounds = torch.tensor(
        [source_w, source_h], dtype=torch.float32, device=depth.device,
    )
    if (pixel.shape != (2,) or not bool(torch.isfinite(pixel).all())
            or not bool(((pixel >= 0) & (pixel < bounds)).all())):
        raise ValueError("subject_pixel is outside the source image.")
    uv = resize_subject_pixel(pixel, source_size, (width, height), depth.device)
    if depth_override is None:
        index = torch.round(uv + 0.5).long() - 1
        index = torch.maximum(index, torch.zeros_like(index))
        index = torch.minimum(index, torch.tensor(
            [width - 1, height - 1], dtype=index.dtype, device=depth.device,
        ))
        u, v = index.tolist()
        patch = depth[max(0, v-sample_radius):min(height, v+sample_radius+1),
                      max(0, u-sample_radius):min(width, u+sample_radius+1)]
        valid = patch[torch.isfinite(patch) & (patch > 0)]
        if valid.numel() == 0:
            raise ValueError("No finite positive depth near the selected subject.")
        subject_depth = valid.float().median()
    else:
        subject_depth = torch.as_tensor(depth_override, dtype=torch.float32, device=depth.device)
    if not bool(torch.isfinite(subject_depth) and subject_depth > 0):
        raise ValueError("Subject depth must be finite and positive.")
    if not bool(torch.isfinite(uv).all()):
        raise ValueError("Subject pixel mapping is not finite.")
    homogeneous = torch.cat([uv + 0.5, uv.new_ones(1)])
    ray = torch.linalg.solve(intrinsic.float(), homogeneous)
    camera_point = ray * subject_depth
    world_point = c2w[:3, :3].float() @ camera_point + c2w[:3, 3].float()
    return {
        "source_pixel": uv.new_tensor(pixel),
        "resized_pixel": uv,
        "depth": subject_depth,
        "camera_point": camera_point,
        "world_point": world_point,
    }


def look_at_rotation(position, target, world_up, initial_rotation=None):
    """OpenCV camera-to-world rotation, evaluated on the input device."""
    forward = target - position
    if float(torch.linalg.vector_norm(forward)) < 1e-8:
        raise ValueError("Camera position coincides with the subject; use a positive d_r.")
    if not bool(torch.isfinite(torch.cat([position, target, world_up])).all()) or float(torch.linalg.vector_norm(world_up)) < 1e-8:
        raise ValueError("Look-at inputs must be finite and world_up nonzero.")
    forward = forward / torch.linalg.vector_norm(forward)
    up = world_up / torch.linalg.vector_norm(world_up)
    right = torch.cross(forward, up, dim=0)
    if float(torch.linalg.vector_norm(right)) < 1e-6:
        fallback = None
        if initial_rotation is not None:
            candidate = initial_rotation[:3, 0].float()
            candidate = candidate - forward * torch.dot(candidate, forward)
            if float(torch.linalg.vector_norm(candidate)) >= 1e-6:
                fallback = candidate
        if fallback is None:
            fallback = torch.tensor([1., 0., 0.], dtype=forward.dtype, device=forward.device)
            if abs(float(torch.dot(forward, fallback))) > 0.9:
                fallback = torch.tensor([0., 1., 0.], dtype=forward.dtype, device=forward.device)
        right = torch.cross(forward, fallback, dim=0)
    right = right / torch.linalg.vector_norm(right)
    down = torch.cross(forward, right, dim=0)
    down = down / torch.linalg.vector_norm(down)
    return torch.stack([right, down, forward], dim=1)


def rotation_to_quaternion(matrix):
    values = torch.stack([
        torch.stack([1+matrix[0, 0]+matrix[1, 1]+matrix[2, 2], matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0], matrix[1, 0]-matrix[0, 1]]),
        torch.stack([matrix[2, 1]-matrix[1, 2], 1+matrix[0, 0]-matrix[1, 1]-matrix[2, 2], matrix[0, 1]+matrix[1, 0], matrix[0, 2]+matrix[2, 0]]),
        torch.stack([matrix[0, 2]-matrix[2, 0], matrix[0, 1]+matrix[1, 0], 1-matrix[0, 0]+matrix[1, 1]-matrix[2, 2], matrix[1, 2]+matrix[2, 1]]),
        torch.stack([matrix[1, 0]-matrix[0, 1], matrix[0, 2]+matrix[2, 0], matrix[1, 2]+matrix[2, 1], 1-matrix[0, 0]-matrix[1, 1]+matrix[2, 2]]),
    ])
    index = torch.argmax(torch.diagonal(values))
    q = values[index] / (2 * values[index, index].clamp_min(1e-16).sqrt())
    return q / torch.linalg.vector_norm(q)


def quaternion_to_rotation(q):
    w, x, y, z = q.unbind()
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)]),
        torch.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)]),
        torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]),
    ])


def slerp(q0, q1, alpha):
    if float(torch.dot(q0, q1)) < 0:
        q1 = -q1
    cosine = torch.dot(q0, q1).clamp(-1, 1)
    if float(cosine) > 0.9995:
        q = q0 + alpha * (q1-q0)
        return q / torch.linalg.vector_norm(q)
    angle = torch.acos(cosine)
    return (torch.sin((1-alpha)*angle)*q0 + torch.sin(alpha*angle)*q1) / torch.sin(angle)


def aim_cameras_at_subject(c2ws, subject_world, world_up=None, ramp_fraction=0.1):
    if not isinstance(ramp_fraction, (int, float)) or not 0 <= ramp_fraction <= 1:
        raise ValueError("subject_look_at_start must be between 0 and 1.")
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4) or len(c2ws) == 0:
        raise ValueError("c2ws must be a nonempty [N,4,4] tensor.")
    if not bool(torch.isfinite(c2ws).all()) or subject_world.shape != (3,) or not bool(torch.isfinite(subject_world).all()):
        raise ValueError("Camera trajectory and subject must be finite.")
    poses = c2ws.float().clone()
    if world_up is None:
        world_up = -poses[0, :3, 1]
    q0 = rotation_to_quaternion(poses[0, :3, :3])
    for i in range(1, len(poses)):
        desired = look_at_rotation(
            poses[i, :3, 3], subject_world, world_up,
            initial_rotation=poses[0, :3, :3],
        )
        alpha = 1.0 if ramp_fraction == 0 else min(i / max(len(poses)-1, 1) / ramp_fraction, 1.0)
        rotation = quaternion_to_rotation(slerp(q0, rotation_to_quaternion(desired), alpha))
        poses[i, :3, :3] = rotation
    return poses.to(c2ws.dtype)
