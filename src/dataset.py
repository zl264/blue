import json

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from src.camera import get_camera_embedding
from src.utils import load_video


def load_dataset(reference_image, render_path, nframe, max_area, pipe, use_camera_embedding,
                 device, sp_degree=1, logger=None, load_human_info=False):
    image = Image.open(reference_image)
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    mod_value = mod_value * sp_degree
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    if logger is not None:
        logger.info(f"Resized image to {height}x{width}")
    image = image.resize((width, height))

    # load conditions
    render_frames = load_video(f"{render_path}/render.mp4")
    if len(render_frames) != nframe:
        raise ValueError(f"render.mp4: expected {nframe} frames, got {len(render_frames)}")
    render_video = torch.stack([ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
    # replace the first frame with the high-quality reference image
    render_video[0, 0] = ToTensor()(image) * 2 - 1
    render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
    render_mask_frames = load_video(f"{render_path}/render_mask.mp4")
    if len(render_mask_frames) != nframe:
        raise ValueError(f"render_mask.mp4: expected {nframe} frames, got {len(render_mask_frames)}")
    render_mask = torch.stack([ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
    render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]  # [0,1],[1,f,1,h,w]
    render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1

    # load camera
    with open(f"{render_path}/cam_info.json", encoding="utf-8") as reader:
        cam_info = json.load(reader)
    render_height, render_width = render_frames[0].height, render_frames[0].width
    cam_info.setdefault("width", render_width)
    cam_info.setdefault("height", render_height)
    if (cam_info["width"], cam_info["height"]) != (render_width, render_height):
        raise ValueError("cam_info.json dimensions must match the actual rendered frames")
    if len(cam_info["extrinsic"]) != nframe:
        raise ValueError(f"cam_info.json: expected {nframe} camera poses")
    geometry_device = torch.device("cpu") if torch.device(device).type == "npu" else device
    w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=geometry_device)
    intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=geometry_device)
    intrinsic[0, :] = intrinsic[0, :] / cam_info["width"] * width
    intrinsic[1, :] = intrinsic[1, :] / cam_info["height"] * height
    if "per_frame_intrinsic" in cam_info:
        # New render outputs may carry a focal-length trajectory. Preserve the
        # original single-K behavior for older outputs.
        intrinsic = torch.tensor(np.array(cam_info["per_frame_intrinsic"]), dtype=torch.float32, device=geometry_device)
        intrinsic[:, 0, :] = intrinsic[:, 0, :] / cam_info["width"] * width
        intrinsic[:, 1, :] = intrinsic[:, 1, :] / cam_info["height"] * height
        if intrinsic.shape[0] != nframe:
            raise ValueError("per_frame_intrinsic must contain one K per rendered frame.")
    else:
        intrinsic = intrinsic[None].repeat(nframe, 1, 1)
    if use_camera_embedding:
        camera_embedding = get_camera_embedding(intrinsic, w2cs, nframe, height, width, normalize=True)
    else:
        camera_embedding = None

    if not load_human_info:
        return image, render_video, render_mask, camera_embedding, height, width
    else:
        smpl_frames = load_video(f"{render_path}/smpl_render.mp4")
        smpl_video = torch.stack([ToTensor()(frame) for frame in smpl_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        smpl_video = F.interpolate(smpl_video, size=(height, width), mode='bicubic')[None]
        smpl_video = torch.clip(smpl_video, -1, 1)  # [-1~1], [1,f,c,h,w]
        hand_frames = load_video(f"{render_path}/hand_render.mp4")
        hand_video = torch.stack([ToTensor()(frame) for frame in hand_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        hand_video = F.interpolate(hand_video, size=(height, width), mode='bicubic')[None]
        hand_video = torch.clip(hand_video, -1, 1)  # [-1~1], [1,f,c,h,w]

        return image, render_video, render_mask, camera_embedding, smpl_video, hand_video, height, width
