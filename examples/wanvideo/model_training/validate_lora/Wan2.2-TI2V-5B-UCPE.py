import argparse
import json

import torch
from PIL import Image

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.models.wan_video_ucpe import LoadUCPECameraTrajectory
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video


def parse_model_paths(paths):
    return [ModelConfig(path=path) for path in json.loads(paths)] if paths else None


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V-5B I2V inference with additive UCPE ray conditioning.")
    parser.add_argument("--checkpoint", required=True, help="Training output containing DiT LoRA and ucpe_camera_encoder.* tensors.")
    parser.add_argument("--camera_trajectory", required=True, help="JSON/JSONL/NPZ/RealEstate10K camera trajectory.")
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    parser.add_argument("--model_paths", default=None, help='Optional local model paths as JSON, e.g. ["models/dit.safetensors","models/t5.pth","models/vae.pth"].')
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output", default="wan22_ucpe_i2v.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--world_coordinates", action="store_true", help="Must match training; default uses first-camera-relative coordinates.")
    parser.add_argument("--translation_scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.model_paths is None:
        model_configs = [
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ]
    else:
        model_configs = parse_model_paths(args.model_paths)
    tokenizer_config = (
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
        if args.tokenizer_path is None else ModelConfig(path=args.tokenizer_path)
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    state_dict = load_state_dict(args.checkpoint)
    camera_prefix = next(
        (prefix for prefix in ("ucpe_camera_encoder.", "pipe.dit.ucpe_camera_encoder.")
         if prefix + "norm.weight" in state_dict),
        None,
    )
    if camera_prefix is None:
        raise ValueError(f"{args.checkpoint} has no UCPE camera encoder weights; this is not a UCPE camera checkpoint.")
    in_dim = state_dict[camera_prefix + "norm.weight"].shape[0]
    hidden_dim = state_dict[camera_prefix + "input_projection.weight"].shape[0]
    condition = {16: "relray", 19: "relray_absmap"}.get(in_dim)
    if condition is None:
        raise ValueError(f"Unrecognized UCPE ray feature dimension: {in_dim}")
    pipe.enable_ucpe_camera(
        condition=condition, hidden_dim=hidden_dim,
        normalize_to_first=not args.world_coordinates,
        translation_scale=args.translation_scale,
    )
    pipe.load_ucpe_camera_checkpoint(state_dict)
    pipe.load_lora(pipe.dit, state_dict=state_dict, alpha=1)
    pipe.dit.ucpe_camera_encoder.eval()

    trajectory = LoadUCPECameraTrajectory()(args.camera_trajectory)
    image = ImageCropAndResize(height=args.height, width=args.width)(Image.open(args.input_image).convert("RGB"))
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=image,
        camera_trajectory=trajectory,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        tiled=args.tiled,
    )
    save_video(video, args.output, fps=args.fps, quality=5)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
