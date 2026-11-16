"""Subject lifting and OpenCV look-at geometry without renderer dependencies.

Pixel coordinates follow image conventions: x right, y down. Depth is positive
camera-z in the same unit as camera translations; poses are camera-to-world.
"""
import numpy as np


def resize_subject_pixel(pixel, source_size, target_size):
    """Map a point through the same non-cropped resize as the reference RGB."""
    source_w, source_h = source_size
    target_w, target_h = target_size
    if min(source_w, source_h, target_w, target_h) <= 0:
        raise ValueError("Image sizes must be positive.")
    u, v = np.asarray(pixel, dtype=np.float64)
    if not np.isfinite([u, v]).all() or not (0 <= u < source_w and 0 <= v < source_h):
        raise ValueError(f"subject_pixel {(u, v)} is outside source image {(source_w, source_h)}.")
    return np.array([
        (u + 0.5) * target_w / source_w - 0.5,
        (v + 0.5) * target_h / source_h - 0.5,
    ])


def lift_subject_to_world(pixel, depth, intrinsic, c2w_0, source_size,
                          depth_override=None, sample_radius=2):
    """Lift a source-image click using RGB-aligned depth and the first camera.

    Returns a metadata dictionary. The resized depth is sampled with a small
    robust median patch by default. ``depth_override`` is already in rendered
    depth units and is never multiplied by an image-resize factor.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 4 and depth.shape[:2] == (1, 1):
        depth = depth[0, 0]
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H,W] or [1,1,H,W].")
    target_h, target_w = depth.shape
    resized_pixel = resize_subject_pixel(pixel, source_size, (target_w, target_h))
    if not isinstance(sample_radius, int) or sample_radius < 0:
        raise ValueError("subject_depth_radius must be a nonnegative integer.")

    if depth_override is None:
        u, v = np.rint(resized_pixel + 0.5).astype(int) - 1
        u, v = np.clip([u, v], [0, 0], [target_w - 1, target_h - 1])
        patch = depth[max(0, v-sample_radius):min(target_h, v+sample_radius+1),
                      max(0, u-sample_radius):min(target_w, u+sample_radius+1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            raise ValueError("No finite positive depth near the selected subject.")
        subject_depth = float(np.median(valid))
    else:
        subject_depth = float(depth_override)
    if not np.isfinite(subject_depth) or subject_depth <= 0:
        raise ValueError("Subject depth must be finite and positive.")

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    c2w_0 = np.asarray(c2w_0, dtype=np.float64)
    if intrinsic.shape != (3, 3) or c2w_0.shape != (4, 4):
        raise ValueError("intrinsic and c2w_0 must have shapes [3,3] and [4,4].")
    if not np.isfinite(intrinsic).all() or not np.isfinite(c2w_0).all():
        raise ValueError("Camera calibration contains NaN or infinity.")
    ray = np.linalg.solve(intrinsic, np.array([resized_pixel[0]+0.5, resized_pixel[1]+0.5, 1.0]))
    point_camera = ray * subject_depth
    point_world = c2w_0[:3, :3] @ point_camera + c2w_0[:3, 3]
    return {
        "source_pixel": np.asarray(pixel, dtype=np.float64),
        "resized_pixel": resized_pixel,
        "depth": subject_depth,
        "camera_point": point_camera,
        "world_point": point_world,
        "source_size": tuple(source_size),
        "render_size": (target_w, target_h),
        "depth_override": depth_override is not None,
    }


def look_at_rotation(camera_position, target, world_up=(0, -1, 0), initial_rotation=None):
    """Return an OpenCV c2w rotation (x right, y down, z forward).

    The default up is image-up (-y). A perpendicular fallback is selected when
    up and the viewing direction are parallel, avoiding NaNs at zenith views.
    """
    position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    if position.shape != (3,) or target.shape != (3,) or up.shape != (3,):
        raise ValueError("Camera position, target and world_up must be 3D vectors.")
    forward = target - position
    if not np.isfinite([*position, *target, *up]).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("Look-at inputs must be finite and world_up must be nonzero.")
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("Camera position coincides with the subject; use a positive d_r.")
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        fallback = None
        if initial_rotation is not None:
            # Preserve the source right axis near zenith, so opposite sides of
            # an elevation orbit cannot cause a discontinuous 180-degree roll.
            fallback = np.asarray(initial_rotation, dtype=np.float64)[:3, 0]
            fallback = fallback - forward * np.dot(fallback, forward)
            if np.linalg.norm(fallback) < 1e-6:
                fallback = None
        if fallback is None:
            fallback = np.array([1., 0., 0.])
            if abs(np.dot(forward, fallback)) > 0.9:
                fallback = np.array([0., 1., 0.])
        right = np.cross(forward, fallback)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.stack([right, down, forward], axis=1)


def _rotation_to_quaternion(matrix):
    """Numerically stable scalar-first quaternion, including 180-degree turns."""
    m = np.asarray(matrix, dtype=np.float64)
    candidates = np.empty((4, 4))
    candidates[0] = [1+m[0,0]+m[1,1]+m[2,2], m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]]
    candidates[1] = [m[2,1]-m[1,2], 1+m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0]]
    candidates[2] = [m[0,2]-m[2,0], m[0,1]+m[1,0], 1-m[0,0]+m[1,1]-m[2,2], m[1,2]+m[2,1]]
    candidates[3] = [m[1,0]-m[0,1], m[0,2]+m[2,0], m[1,2]+m[2,1], 1-m[0,0]-m[1,1]+m[2,2]]
    index = int(np.argmax(np.diag(candidates)))
    q = candidates[index]
    q = q / (2 * np.sqrt(max(candidates[index, index], 1e-16)))
    return q / np.linalg.norm(q)


def _quaternion_to_rotation(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _slerp(q0, q1, alpha):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    cosine = np.clip(np.dot(q0, q1), -1, 1)
    if cosine > 0.9995:
        q = q0 + alpha * (q1-q0)
        return q / np.linalg.norm(q)
    angle = np.arccos(cosine)
    return (np.sin((1-alpha)*angle)*q0 + np.sin(alpha*angle)*q1) / np.sin(angle)


def aim_cameras_at_subject(c2ws, subject_world, world_up=None, ramp_fraction=0.1):
    """Aim at a world point while preserving frame 0 exactly.

    Each later camera gets a look-at rotation. A short geodesic interpolation
    avoids an abrupt rotation between the source frame and frame 1. With
    ``ramp_fraction=0``, every frame after frame 0 is exactly aimed at subject.
    """
    poses = np.array(c2ws, dtype=np.float64, copy=True)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("c2ws must be a nonempty [N,4,4] array.")
    if not np.isfinite(ramp_fraction) or not 0 <= ramp_fraction <= 1:
        raise ValueError("subject_look_at_start must be between 0 and 1.")
    if not np.isfinite(poses).all():
        raise ValueError("Camera trajectory contains NaN or infinity.")
    target = np.asarray(subject_world, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("subject_world must be a finite 3D point.")
    if world_up is None:
        # Preserve the initial camera roll/elevation reference by default.
        world_up = -poses[0, :3, 1]
    q0 = _rotation_to_quaternion(poses[0, :3, :3])
    denominator = max(len(poses)-1, 1)
    for i in range(1, len(poses)):
        desired = look_at_rotation(poses[i, :3, 3], target, world_up, initial_rotation=poses[0, :3, :3])
        alpha = 1.0 if ramp_fraction == 0 else min(i / denominator / ramp_fraction, 1.0)
        rotation = _quaternion_to_rotation(_slerp(q0, _rotation_to_quaternion(desired), alpha))
        poses[i, :3, :3] = rotation
    return poses
