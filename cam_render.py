import argparse
import json
import os
import warnings

import einops
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from diffusers.utils import export_to_video
from torchvision.transforms import ToTensor, ToPILImage

from src.depth_estimation import create_depth_estimator
from src.accelerator import empty_cache, setup_device
from src.pointcloud import point_rendering, contract_depth
from src.utils import traj_map, points_padding, set_initial_camera, build_cameras
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


def create_foreground_segmenter(device, model_path):
    """Keep CarveKit's official CUDA wrapper, use direct NPU inference otherwise."""
    if device.type == "cuda":
        from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7

        return TracerUniversalB7(device=device.type, batch_size=1, model_path=model_path).eval()

    from carvekit.ml.arch.tracerb7.tracer import TracerDecoder
    from carvekit.ml.arch.tracerb7.efficientnet import EfficientEncoderB7

    class DeviceTracerB7(TracerDecoder):
        def __init__(self):
            super().__init__(
                encoder=EfficientEncoderB7(), rfb_channel=[32, 64, 128],
                features_channels=[48, 80, 224, 640],
            )
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=False)
            self.to(device=device, dtype=torch.float32).eval()

        def __call__(self, images):
            if len(images) != 1:
                raise ValueError("Static preprocessing currently uses one image per call")
            from torchvision.transforms import Compose, Normalize, Resize

            image = images[0].convert("RGB")
            original_size = image.size
            transform = Compose([
                ToTensor(), Resize((640, 640)),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            batch = transform(image)[None].to(device=device, dtype=torch.float32)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                output = super().__call__(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                mask = torch.nn.functional.interpolate(
                    output.float(), size=(original_size[1], original_size[0]),
                    mode="bilinear", align_corners=False,
                )[0, 0].clamp(0, 1).cpu().numpy()
            return [Image.fromarray((mask * 255).round().astype(np.uint8), mode="L")]

    return DeviceTracerB7()


def make_raster_settings(height, width, device):
    if device.type in ("npu", "cpu"):
        # The pure PyTorch fallback consumes explicit splat/chunk options instead
        # of a PyTorch3D settings object. Preserve the API for alignment.py.
        return None
    from pytorch3d.renderer import PointsRasterizationSettings

    return PointsRasterizationSettings(
        image_size=(height, width), radius=0.008, points_per_pixel=8,
    )


def main():
    torch.set_grad_enabled(False)
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", default=None, type=str, required=True,
                        help="the path of input image")
    parser.add_argument("--output_path", default="outputs/temp", type=str,
                        help="output folder's path")
    parser.add_argument("--traj_type", default="custom", type=str,
                        choices=["custom", "free1", "free2", "free3", "free4", "free5", "swing1", "swing2", "orbit"],
                        help="custom refers to a custom trajectory, while the others are pre-defined camera trajectories (see traj_map for details)")
    parser.add_argument("--nframe", default=81, type=int, help="Total number of frames")
    parser.add_argument("--d_r", default=1.0, type=float,
                        help="Camera distance, default is 1.0, range 0.25 to 2.5")
    parser.add_argument("--d_theta", default=0.0, type=float,
                        help="Vertical rotation, <0 up, >0 down, range -90 to 30; generally not recommended to angle too much downwards")
    parser.add_argument("--d_phi", default=0.0, type=float,
                        help="Horizontal rotation, <0 right, >0 left, supports 360 degrees; range -360 to 360")
    parser.add_argument("--x_offset", default=0.0, type=float,
                        help="Horizontal translation, <0 left, >0 right, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--y_offset", default=0.0, type=float,
                        help="Vertical translation, <0 up, >0 down, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--z_offset", default=0.0, type=float,
                        help="Forward and backward translation, <0 back, >0 forward, range -0.5 to 0.5 is ok; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--focal_length", default=1.0, type=float,
                        help="Focal length, range 0.25 to 2.5; changing focal length zooms in and out")
    parser.add_argument("--start_elevation", default=5.0, type=float,
                        help="Initial angle, no exceptions to change")
    parser.add_argument("--depth_model", default="moge3-vitl",
                        choices=["moge3-vitl", "moge3-vitg", "moge2-vitl", "depth-pro"],
                        help="Metric depth backend; MoGe-3 weights are MIT, use depth-pro to reproduce the original pipeline")
    parser.add_argument("--depth_checkpoint", default=None, type=str,
                        help="Optional local/Hugging Face MoGe checkpoint; defaults to the official selected variant")
    parser.add_argument("--depth_refine_steps", default=3, type=int,
                        help="MoGe-3 sparse volumetric refinement steps; 3 follows the official default")
    parser.add_argument("--depth_fp16", action="store_true",
                        help="Use MoGe mixed precision; validate geometry before production use")
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"],
                        help="Use npu for depth, segmentation and pure-PyTorch point rendering on Ascend")
    parser.add_argument("--renderer_backend", default="auto", choices=["auto", "pytorch3d", "torch"],
                        help="auto selects pure PyTorch on NPU/CPU and PyTorch3D on CUDA")
    parser.add_argument("--torch_splat_radius", default=1, type=int,
                        help="Pure-PyTorch output-pixel splat radius; 0 is nearest-point, 1 covers a 3x3 footprint")
    parser.add_argument("--torch_chunk_size", default=32768, type=int,
                        help="Source points per rendering pass; reduce if preprocessing runs out of memory")
    parser.add_argument("--subject_pixel", default=None, type=float, nargs=2, metavar=("U", "V"),
                        help="Subject pixel [u,v] in the EXIF-corrected original reference image; enables subject-centered camera motion")
    parser.add_argument("--subject_depth", default=None, type=float,
                        help="Optional positive camera-z override at subject_pixel, in the selected backend's rendered metric depth units; otherwise sample the aligned depth map")
    parser.add_argument("--subject_depth_radius", default=2, type=int,
                        help="Median depth patch radius in resized pixels; 0 samples only the nearest pixel")
    parser.add_argument("--subject_world_up", default=[0.0, -1.0, 0.0], type=float, nargs=3,
                        help="World-space up direction for look-at; default is camera/world -y")
    parser.add_argument("--subject_look_at_start", default=0.1, type=float,
                        help="Fraction of trajectory spent smoothly turning from the original view to subject-centered look-at; 0 turns immediately after frame 0")
    parser.add_argument("--subject_keep_upright", action="store_true",
                        help="Use subject_world_up (default world -y) instead of retaining the source camera's up direction")

    args = parser.parse_args()
    if args.nframe < 1 or args.nframe % 4 != 1:
        raise ValueError("Wan temporal VAE requires nframe % 4 == 1")
    if args.depth_refine_steps < 0:
        raise ValueError("depth_refine_steps must be nonnegative.")
    device = setup_device(args.device)
    if args.renderer_backend == "pytorch3d" and device.type in ("npu", "cpu"):
        raise ValueError("PyTorch3D rendering is not available in the NPU/CPU static path; use --renderer_backend torch.")
    if args.torch_splat_radius < 0 or args.torch_chunk_size <= 0:
        raise ValueError("torch_splat_radius must be nonnegative and torch_chunk_size positive")
    print(f"Init depth model: {args.depth_model} on {device}")
    depth_estimator = create_depth_estimator(
        args.depth_model, device=device, checkpoint=args.depth_checkpoint,
        refine_steps=args.depth_refine_steps, use_fp16=args.depth_fp16,
    )

    print("Init mask model")

    # == motion definition ==
    if args.traj_type == "custom":
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = \
            "free", args.x_offset, args.y_offset, args.z_offset, args.d_theta, args.d_phi, args.d_r
    else:
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(args.traj_type)

    # load image
    image = Image.open(args.reference_image).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w_origin, h_origin = image.size
    hw_list = [[480, 768], [512, 720], [608, 608], [720, 512], [768, 480]]
    hw_ratio_list = [h_ / w_ for [h_, w_] in hw_list]
    hw_ratio = h_origin / w_origin
    sub_hw_ratio = np.abs(np.array(hw_ratio_list) - hw_ratio)
    select_idx = np.argmin(sub_hw_ratio)
    height, width = hw_list[select_idx]
    print(f"Image: {args.reference_image.split('/')[-1]}, Resolution: {h_origin}x{w_origin}->{height}x{width}")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1

    # All depth backends see the same resized RGB pixel grid. MoGe consumes
    # [0,1] RGB directly; Depth-Pro owns its own normalization in its adapter.
    depth_prediction = depth_estimator.predict(validation_image[0])
    depth = depth_prediction.depth[None, None].to(device)
    K = depth_prediction.intrinsic.float().to(device)
    K_inv = K.inverse()
    intrinsic = K[None].repeat(args.nframe, 1, 1)
    depth_image = validation_image[0].to(device) * 2 - 1  # Colors for exported point cloud.
    print(f"Depth backend: {depth_prediction.model_name}; "
          f"invalid pixels conservatively filled: {depth_prediction.filled_pixels}; "
          f"fx/fy: {K[0,0].item():.3f}/{K[1,1].item():.3f}")
    del depth_estimator
    empty_cache(device)

    os.makedirs(args.output_path, exist_ok=True)

    # Load the segmentation network only after releasing the depth backend.
    print("Init mask model")
    seg_net = create_foreground_segmenter(
        device, hf_hub_download(repo_id="Carve/tracer_b7", filename="tracer_b7.pth"),
    )

    # get pointcloud
    points2d = torch.stack(torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device), indexing="xy"), -1)
    points3d = points_padding(points2d).reshape(height * width, 3)  # [hw,3]
    points3d = (K_inv @ points3d.T * depth.reshape(1, height * width)).T
    colors = ((depth_image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(height * width, 3)

    # inference foreground mask
    with torch.no_grad():
        origin_w_, origin_h_ = image.size
        image_pil = image.resize((512, 512))
        fg_mask = seg_net([image_pil])[0]
        fg_mask = fg_mask.resize((origin_w_, origin_h_))
    fg_mask = np.array(fg_mask)
    fg_mask = fg_mask > 127.5
    fg_mask = torch.tensor(fg_mask)
    del seg_net
    empty_cache(device)
    if fg_mask.float().mean() < 0.05:
        fg_mask[...] = True
    depth_avg = torch.median(depth[0, 0, fg_mask.to(device)]).item()

    w2c_0, c2w_0 = set_initial_camera(args.start_elevation, depth_avg, device=device)

    # Convert points3d to world coordinates on the selected device. Transfer
    # only the optional PLY export copy to the host.
    points3d_world = (c2w_0[:3] @ points_padding(points3d).T).T
    pcd = trimesh.PointCloud(vertices=points3d_world.cpu().numpy(), colors=colors.cpu().numpy())
    _ = pcd.export(f"{args.output_path}/pcd.ply")

    if device.type in ("npu", "cpu"):
        from src.subject_camera_torch import aim_cameras_at_subject, lift_subject_to_world
    else:
        from src.subject_camera import aim_cameras_at_subject, lift_subject_to_world

    # The original trajectory uses the world origin at foreground depth as its
    # center. Shift the entire orbit to a clicked subject's 3D position instead;
    # the initial camera position and all preset/custom parameters stay intact.
    subject_info = None
    orbit_center = None
    if args.subject_pixel is not None:
        # Use the same pixel-center convention and far-depth contraction as
        # point_rendering, rather than the export-only integer-coordinate path.
        render_depth = depth.detach().clone()
        if render_depth.max() > 0:
            render_depth = contract_depth(render_depth)
        lift_args = (K, c2w_0)
        lift_depth = depth.detach()
        lift_render_depth = render_depth
        if device.type == "cuda":
            # The NumPy geometry helper is retained only for the legacy CUDA path.
            lift_args = (K.cpu().numpy(), c2w_0.cpu().numpy())
            lift_depth = lift_depth.cpu().numpy()
            lift_render_depth = lift_render_depth.cpu().numpy()
        source_subject_info = lift_subject_to_world(
            args.subject_pixel, lift_depth, *lift_args,
            source_size=(w_origin, h_origin), sample_radius=args.subject_depth_radius,
        )
        subject_info = lift_subject_to_world(
            args.subject_pixel, lift_render_depth, *lift_args,
            source_size=(w_origin, h_origin), depth_override=args.subject_depth,
            sample_radius=args.subject_depth_radius,
        )
        subject_info["source_camera_z"] = source_subject_info["depth"]
        orbit_center = subject_info["world_point"]
        print(f"Subject pixel: {subject_info['source_pixel'].tolist()} -> "
              f"{subject_info['resized_pixel'].tolist()}, camera-z: {subject_info['depth']:.6f}")
        print(f"Subject world point: {orbit_center.tolist()}")
        print(f"Subject source camera-z (before renderer far-depth contraction): {source_subject_info['depth']:.6f}")

    # build camera viewpoints according to d_theta，d_phi, d_r
    w2cs, c2ws, intrinsic = build_cameras(cam_traj=cam_traj,
                                          w2c_0=w2c_0,
                                          c2w_0=c2w_0,
                                          intrinsic=intrinsic,
                                          nframe=args.nframe,
                                          focal_length=args.focal_length,
                                          d_theta=d_theta,
                                          d_phi=d_phi,
                                          d_r=d_r,
                                          radius=depth_avg,
                                          x_offset=x_offset,
                                          y_offset=y_offset,
                                          z_offset=z_offset,
                                          orbit_center=orbit_center)
    if subject_info is not None:
        world_up = args.subject_world_up if args.subject_keep_upright else None
        if device.type == "cuda":
            c2ws = torch.from_numpy(aim_cameras_at_subject(
                c2ws.numpy(), subject_info["world_point"], world_up,
                ramp_fraction=args.subject_look_at_start,
            )).to(c2ws.dtype)
        else:
            if world_up is not None:
                world_up = torch.as_tensor(world_up, dtype=c2ws.dtype, device=device)
            c2ws = aim_cameras_at_subject(
                c2ws, subject_info["world_point"], world_up,
                ramp_fraction=args.subject_look_at_start,
            )
        w2cs = torch.linalg.inv(c2ws)
        if not torch.isfinite(w2cs).all():
            raise ValueError("Subject camera trajectory contains NaN or infinity.")
    # save camera infos
    w2cs_list = w2cs.cpu().numpy().tolist()
    camera_infos = {
        "intrinsic": K.cpu().numpy().tolist(),
        # Original single-K field remains for backward-compatible consumers.
        # This optional dense field exactly matches the per-frame rendering K,
        # including the existing focal_length ramp.
        "per_frame_intrinsic": intrinsic.cpu().numpy().tolist(),
        "extrinsic": w2cs_list, "height": height, "width": width,
        "preprocessing": {
            "device": device.type,
            "renderer_backend": (
                "torch" if args.renderer_backend == "auto" and device.type in ("npu", "cpu")
                else "pytorch3d" if args.renderer_backend == "auto" else args.renderer_backend
            ),
            "torch_splat_radius": args.torch_splat_radius,
            "torch_chunk_size": args.torch_chunk_size,
        },
        "depth_model": {
            "name": args.depth_model,
            "checkpoint": args.depth_checkpoint,
            "predicted_name": depth_prediction.model_name,
            "refine_steps": depth_prediction.refine_steps,
            "fp16": depth_prediction.use_fp16,
            "filled_pixels": depth_prediction.filled_pixels,
            "filled_pixel_fraction": depth_prediction.filled_pixels / (height * width),
        },
    }
    if subject_info is not None:
        camera_infos["subject"] = {
            key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor)
            else value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in subject_info.items()
        }
        camera_infos["subject"]["world_up"] = (
            args.subject_world_up if args.subject_keep_upright
            else (-c2w_0.cpu().numpy()[:3, 1]).tolist()
        )
        camera_infos["subject"]["look_at_ramp_fraction"] = args.subject_look_at_start
        camera_infos["subject"]["source_depth"] = camera_infos["subject"]["source_camera_z"]
        camera_infos["subject"]["render_depth"] = camera_infos["subject"]["depth"]
        camera_infos["subject"]["initial_c2w"] = c2w_0.cpu().numpy().tolist()
    with open(f"{args.output_path}/cam_info.json", "w") as writer:
        json.dump(camera_infos, writer, indent=2)
    if subject_info is not None:
        with open(f"{args.output_path}/subject_info.json", "w", encoding="utf-8") as writer:
            json.dump(camera_infos["subject"], writer, indent=2)

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        control_imgs, render_masks = point_rendering(K=intrinsic.float(),
                                                     w2cs=w2cs.float(),
                                                     depth=depth.float(),
                                                     image=(validation_image.float() * 2 - 1).to(device),
                                                     raster_settings=make_raster_settings(height, width, device),
                                                     device=device,
                                                     background_color=[0, 0, 0],
                                                     sobel_threshold=0.35,
                                                     sam_mask=None,
                                                     renderer_backend=args.renderer_backend,
                                                     torch_splat_radius=args.torch_splat_radius,
                                                     torch_chunk_size=args.torch_chunk_size)

    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=args.nframe)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=args.nframe)

    render_video = []
    mask_video = []
    control_imgs = control_imgs.to(torch.float32)
    for i in range(args.nframe):
        img = ToPILImage()((control_imgs[0][:, i] + 1) / 2)
        render_video.append(img)
        mask = ToPILImage()(render_masks[0][:, i])
        mask_video.append(mask)

    export_to_video(render_video, f"{args.output_path}/render.mp4", fps=16)
    export_to_video(mask_video, f"{args.output_path}/render_mask.mp4", fps=16)

    print("Rendering finished.")


if __name__ == "__main__":
    main()
