import argparse
import gc
import os

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from src.accelerator import empty_cache, init_distributed, sequence_parallel_world_size, setup_device
from src.dataset import load_dataset
from src.fsdp import shard_model
from src.models.controlnet import WanAttnProcessorSP, configure_device_attention
from src.models.pcd_controller import PCDController
from src.pipelines.pipeline_pcd import PCDControllerPipeline
from src.utils import create_logger, is_main_process

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", default=None, type=str, required=True, help="the path of input image")
    parser.add_argument("--render_path", default=None, type=str, required=True, help="the path of render folder")
    parser.add_argument("--output_path", default="result.mp4", type=str, help="output path")
    parser.add_argument("--nframe", default=81, type=int, help="Total number of frames")
    parser.add_argument("--fsdp", action="store_true", help="whether to use fsdp to save memory")
    parser.add_argument("--enable_sp", action="store_true", help="whether to use SP inference")
    parser.add_argument("--prompt", default="This video describes a slow and stable camera movement with high quality and high definition.",
                        type=str, help="Prompt of the reference image")
    parser.add_argument("--max_area", default=480 * 768, type=int, help="Total pixel area of height * width")
    parser.add_argument("--seed", default=1024, type=int, help="random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"],
                        help="Use --device npu on Ascend 910B2; auto prefers NPU over CUDA")
    parser.add_argument("--static_lora", default=None, type=str,
                        help="Directory saved by train_static_lora.py, e.g. outputs/static_lora/latest")
    parser.add_argument("--lora_scale", default=1.0, type=float,
                        help="Static-scene LoRA strength; try 0.5-1.0 if identity/geometry degrades")
    args = parser.parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if (args.enable_sp or args.fsdp) and world_size == 1:
        raise ValueError("SP/FSDP require multiple devices; single 910B2 uses CPU offload")
    device = setup_device(args.device, local_rank)
    if world_size > 1:
        if device.type not in ("npu", "cuda"):
            raise ValueError("Multi-device inference requires NPU or CUDA")
        init_distributed(device)

    cfg = OmegaConf.load(hf_hub_download(repo_id="ewrfcas/Uni3C", filename="config.json"))
    # == init logger ==
    logger = create_logger(None)
    logger.info(f"World size: {world_size}")

    if args.enable_sp:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=world_size, ulysses_degree=world_size)

    if dist.is_initialized():
        seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed, src=0)
        args.seed = seed[0]
        print(f"Rank:{local_rank}, seed:{args.seed}. Please make sure that all SP share the same seed.")

    base_model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    logger.info("loading transformer...")
    transformer = PCDController.from_pretrained(base_model_id, subfolder="transformer", controlnet_cfg=cfg.controlnet_cfg, torch_dtype=torch.bfloat16)
    logger.info("loading controlnet...")
    transformer.build_controlnet(model_path="controlnet.pth", logger=logger)
    if args.static_lora:
        from src.training import load_lora
        transformer = load_lora(transformer, args.static_lora)
        logger.info(f"Loaded static-scene LoRA: {args.static_lora}; scale={args.lora_scale}")
    transformer.enable_gradient_checkpointing()
    transformer.controlnet.gradient_checkpointing = True
    if args.enable_sp:  # replace attention_processor for DiT
        transformer.sp_size = sequence_parallel_world_size()
        for layer in transformer.controlnet.controlnet_blocks:
            layer.self_attn.processor.sp_size = sequence_parallel_world_size()
        for block in transformer.blocks:
            block.attn1.set_processor(WanAttnProcessorSP(sp_size=sequence_parallel_world_size()))
            block.attn2.set_processor(WanAttnProcessorSP(sp_size=sequence_parallel_world_size()))
    else:
        configure_device_attention(transformer, device)
    if args.enable_sp and args.fsdp:
        if dist.is_initialized():
            dist.barrier()
        transformer = shard_model(transformer, device_id=local_rank, model_type="wan", use_orig_params=True)
        gc.collect()
        empty_cache(device)
        logger.info("Finish warp DiT for FSDP...")

    text_encoder = UMT5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    if args.enable_sp and args.fsdp:
        if dist.is_initialized():
            dist.barrier()
        text_encoder = shard_model(text_encoder, device_id=local_rank, model_type="t5", use_orig_params=True)
        gc.collect()
        empty_cache(device)
        logger.info("Finish warp T5 for FSDP...")

    image_clip = CLIPVisionModel.from_pretrained(base_model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    if args.enable_sp and args.fsdp:
        if dist.is_initialized():
            dist.barrier()
        image_clip = shard_model(image_clip, device_id=local_rank, model_type="clip", use_orig_params=True)
        gc.collect()
        empty_cache(device)
        logger.info("Finish warp CLIP for FSDP...")

    vae = AutoencoderKLWan.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32)
    if args.enable_sp and args.fsdp:
        vae = vae.to(device)

    pipe = PCDControllerPipeline(
        tokenizer=AutoTokenizer.from_pretrained(base_model_id, subfolder="tokenizer"),
        text_encoder=text_encoder,
        image_encoder=image_clip,
        image_processor=CLIPImageProcessor.from_pretrained(base_model_id, subfolder="image_processor"),
        transformer=transformer,
        vae=vae,
        scheduler=UniPCMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    )

    # CPU offload avoids requiring the full 14B model and encoders on one NPU.
    if not (args.enable_sp or args.fsdp):
        pipe.enable_model_cpu_offload(gpu_id=local_rank, device=device.type)

    image, render_video, render_mask, camera_embedding, height, width = load_dataset(
        reference_image=args.reference_image,
        render_path=args.render_path,
        nframe=args.nframe,
        max_area=args.max_area,
        pipe=pipe,
        use_camera_embedding=cfg.get("camera_embedding", False),
        device=device,
        sp_degree=sequence_parallel_world_size() if args.enable_sp else 1,
        logger=logger
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    output = pipe(
        image=image,
        render_video=render_video.to(device),
        render_mask=render_mask.to(device),
        camera_embedding=camera_embedding.to(device) if camera_embedding is not None else None,
        prompt=(args.prompt),
        negative_prompt="",
        height=height,
        width=width,
        num_frames=args.nframe,
        guidance_scale=5.0,
        generator=gen,
        attention_kwargs={"scale": args.lora_scale} if args.static_lora else None,
    ).frames[0]

    if is_main_process():
        export_to_video(output, args.output_path, fps=16)
