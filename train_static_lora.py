"""Single-device LoRA fine-tuning of Uni3C for static-scene camera motion."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.accelerator import empty_cache, seed_everything, setup_device, synchronize
from src.models.controlnet import configure_device_attention
from src.models.pcd_controller import PCDController
from src.training import (
    add_controlnet_lora, load_lora, move_cache, save_lora, validate_cache,
    validate_cache_batch,
)

class LatentCacheDataset(Dataset):
    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        self.metadata = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
        self.paths = [cache_dir / f"{sample_id}.pt" for sample_id in self.metadata["samples"]]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise ValueError(f"Missing cached samples: {missing[:5]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cache = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        try:
            validate_cache(cache, self.metadata["settings"].get("camera_embedding", True))
        except ValueError as error:
            raise ValueError(f"Invalid cache {self.paths[index]}: {error}") from error
        return cache


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_id", default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
    parser.add_argument("--device", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    parser.add_argument("--target", choices=["controlnet", "all"], default="controlnet",
                        help="all adds Wan attention LoRA and needs more memory")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--resume_lora", default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep_sampling", choices=["uniform", "logit_normal"], default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--validate_only", action="store_true", help="Build model/optimizer, then exit without training")
    return parser.parse_args()


def run(args):
    if args.max_steps <= 0 or args.gradient_accumulation <= 0 or args.batch_size <= 0:
        raise ValueError("steps, batch_size and gradient_accumulation must be positive")
    if args.resume_lora and not Path(args.resume_lora).is_dir():
        raise FileNotFoundError(args.resume_lora)
    device = setup_device(args.device)
    if device.type == "cpu":
        raise RuntimeError("CPU is useful for preprocessing only; 14B training requires npu or cuda")
    seed_everything(args.seed, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dataset = LatentCacheDataset(args.cache_dir)
    cache_metadata_path = Path(args.cache_dir) / "cache_manifest.json"
    if not cache_metadata_path.exists():
        raise ValueError("cache_manifest.json is required to verify cache/model compatibility")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    cache_settings = cache_metadata.get("settings", cache_metadata)
    if cache_settings.get("base_model_id") != args.base_model_id:
        raise ValueError("Training base_model_id differs from the cached encoder/model")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, drop_last=False,
    )
    cfg = OmegaConf.load(hf_hub_download(repo_id="ewrfcas/Uni3C", filename="config.json"))
    print("Loading frozen Wan transformer and official Uni3C controller")
    transformer = PCDController.from_pretrained(
        args.base_model_id, subfolder="transformer", controlnet_cfg=cfg.controlnet_cfg,
        torch_dtype=torch.bfloat16,
    )
    transformer.build_controlnet(model_path="controlnet.pth")
    if args.resume_lora:
        transformer = load_lora(transformer, args.resume_lora, trainable=True)
    else:
        transformer = add_controlnet_lora(transformer, args.lora_rank, args.lora_alpha, args.target)
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable or not all("lora_" in name for name, parameter in transformer.named_parameters()
                                if parameter.requires_grad):
        raise RuntimeError("No trainable LoRA parameters")
    # build_controlnet creates the new controller modules after from_pretrained;
    # keep those FP32 while the Wan base weights remain BF16.
    transformer = transformer.to(device=device).train()
    configure_device_attention(transformer, device)
    # PEFT's Linear implementation casts small LoRA activations to FP32 adapter
    # dtype and the result back to the base activation dtype. AdamW therefore
    # retains FP32 master parameters without promoting Wan's large activations.
    for parameter in trainable:
        parameter.data = parameter.data.float()
    for module in transformer.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if not args.no_gradient_checkpointing:
        # diffusers 0.33.1 uses use_reentrant=False, so frozen conditioning
        # tensors need not be marked requires_grad for LoRA backpropagation.
        transformer.enable_gradient_checkpointing()
    transformer.controlnet.gradient_checkpointing = not args.no_gradient_checkpointing
    # Wan uses this flag when deciding whether checkpointing is active.
    transformer.gradient_checkpointing = not args.no_gradient_checkpointing
    for module in transformer.modules():
        if hasattr(module, "gradient_checkpointing") and module is not transformer:
            module.gradient_checkpointing = not args.no_gradient_checkpointing

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95), eps=1e-8, fused=False)
    # The official Uni3C scheduler config carries flow_shift=5, not the Euler
    # default shift=1. Preserve that schedule for training noise and timesteps.
    inference_scheduler = UniPCMultistepScheduler.from_pretrained(args.base_model_id, subfolder="scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler(shift=inference_scheduler.config.flow_shift)
    scheduler.set_timesteps(1000, device="cpu")
    sigmas = scheduler.sigmas[:-1].float().cpu()
    timesteps = scheduler.timesteps.float().cpu()
    if sigmas.numel() != 1000:
        raise RuntimeError(f"Expected 1000 training sigmas, got {sigmas.numel()}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = vars(args).copy()
    metadata["trainable_parameters"] = sum(p.numel() for p in trainable)
    metadata["num_samples"] = len(dataset)
    metadata["cache_settings"] = cache_settings
    metadata["flow_shift"] = float(inference_scheduler.config.flow_shift)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as writer:
        json.dump(metadata, writer, indent=2)
    if args.validate_only:
        validation_batch = move_cache(next(iter(loader)), torch.device("cpu"))
        validate_cache_batch(validation_batch)
        print(f"Validation only: {len(dataset)} caches; {metadata['trainable_parameters']:,} trainable parameters; "
              "one cached batch passed shape checks (no forward/backward run)")
        return

    data_iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    running_loss = 0.0
    for step in range(1, args.max_steps + 1):
        for accumulation_step in range(args.gradient_accumulation):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                batch = next(data_iterator)
            batch = move_cache(batch, device)
            validate_cache_batch(batch)
            camera_embedding = batch.get("camera_embedding")
            if isinstance(camera_embedding, torch.Tensor):
                camera_embedding = camera_embedding.float()
            else:
                camera_embedding = None
            target = batch["target_latent"].float()
            noise = torch.randn(target.shape, device=device, dtype=torch.float32)
            if args.timestep_sampling == "uniform":
                indices = torch.randint(0, 1000, (target.shape[0],), generator=generator)
            else:
                normal = torch.randn(target.shape[0], generator=generator) * args.logit_std + args.logit_mean
                indices = (normal.sigmoid() * 999).long().clamp(0, 999)
            sigma = sigmas[indices].to(device)[:, None, None, None, None]
            noisy = (1 - sigma) * target + sigma * noise
            flow_target = noise - target
            model_input = torch.cat([noisy, batch["condition"].float()], dim=1).to(torch.bfloat16)
            timestep = timesteps[indices].to(device=device, dtype=torch.float32)
            with torch.enable_grad():
                prediction = transformer(
                    hidden_states=model_input, render_latent=batch["render_latent"].float(),
                    render_mask=batch["render_mask"].float(), camera_embedding=camera_embedding,
                    timestep=timestep, encoder_hidden_states=batch["prompt_embeds"].to(torch.bfloat16),
                    encoder_hidden_states_image=batch["image_embeds"].to(torch.bfloat16),
                    return_dict=False,
                )[0].float()
                loss = F.mse_loss(prediction, flow_target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
                (loss / args.gradient_accumulation).backward()
            running_loss += loss.item() / args.gradient_accumulation
            del batch, target, noise, noisy, flow_target, model_input, prediction, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        elapsed = time.monotonic() - start
        print(f"step {step}/{args.max_steps} loss={running_loss:.6f} grad_norm={float(grad_norm):.4f} "
              f"elapsed={elapsed:.1f}s steps/s={step / elapsed:.4f}", flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_lora(transformer, output_dir / f"step-{step:06d}",
                      {**metadata, "step": step, "loss": running_loss, "elapsed_seconds": elapsed})
            save_lora(transformer, output_dir / "latest",
                      {**metadata, "step": step, "loss": running_loss, "elapsed_seconds": elapsed})
        running_loss = 0.0
        gc.collect()
        empty_cache(device)


if __name__ == "__main__":
    run(parse_args())
