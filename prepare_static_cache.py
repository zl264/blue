"""Cache frozen Wan/Uni3C encoders for low-memory static-scene fine-tuning."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import einops
import torch
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from src.accelerator import empty_cache, setup_device
from src.dataset import load_dataset
from src.pipelines.pipeline_pcd import PCDControllerPipeline
from src.training import load_manifest, normalize_vae_latent, validate_cache
from src.utils import load_video


def video_tensor(path, height, width, expected_frames):
    frames = load_video(path)
    if len(frames) != expected_frames:
        raise ValueError(f"{path}: expected {expected_frames} frames, got {len(frames)}; trim/split before caching")
    frames = [frame.convert("RGB").resize((width, height), Image.Resampling.BICUBIC) for frame in frames]
    video = torch.stack([ToTensor()(frame) for frame in frames]) * 2 - 1
    return einops.rearrange(video, "f c h w -> 1 c f h w").contiguous()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_signature(sample):
    return {
        "sample_id": sample["sample_id"],
        "prompt": sample["prompt"],
        "reference_image": str(Path(sample["reference_image"]).resolve()),
        "render_path": str(Path(sample["render_path"]).resolve()),
        "target_video": str(Path(sample["target_video"]).resolve()),
        "files": {
            "reference_image": file_sha256(sample["reference_image"]),
            "target_video": file_sha256(sample["target_video"]),
            "render.mp4": file_sha256(Path(sample["render_path"]) / "render.mp4"),
            "render_mask.mp4": file_sha256(Path(sample["render_path"]) / "render_mask.mp4"),
            "cam_info.json": file_sha256(Path(sample["render_path"]) / "cam_info.json"),
        },
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL; see docs/ascend_static_scene.md")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_id", default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
    parser.add_argument("--device", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    parser.add_argument("--nframe", type=int, default=17, help="Start with 17; must equal rendered and target frames")
    parser.add_argument("--max_area", type=int, default=480 * 768)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--camera_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.nframe < 1 or args.nframe % 4 != 1:
        raise ValueError("Wan temporal VAE requires nframe % 4 == 1")
    device = setup_device(args.device)
    dtype = getattr(torch, args.cache_dtype)
    cache_marker = {"nframe": args.nframe, "max_area": args.max_area,
                    "base_model_id": args.base_model_id, "cache_dtype": args.cache_dtype,
                    "camera_embedding": args.camera_embedding,
                    "negative_prompt": args.negative_prompt, "version": 1}
    samples = load_manifest(args.manifest)
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id must be unique")
    sample_signatures = [sample_signature(sample) for sample in samples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / ".partial"
    metadata_path = output_dir / "cache_manifest.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_settings = previous.get("settings", previous)
        if previous_settings != cache_marker:
            raise ValueError("Cache settings differ from cache_manifest.json; use a new output_dir")
        if set(previous.get("samples", [])) != set(sample_ids):
            raise ValueError("Manifest samples differ from cache_manifest.json; use a new output_dir")
        if previous.get("sample_signatures") != sample_signatures:
            raise ValueError("Input files/prompts changed; use a new output_dir or rebuild into an empty directory")

    def needs_encoding(sample):
        path = output_dir / f"{sample['sample_id']}.pt"
        if path.exists():
            try:
                validate_cache(torch.load(path, map_location="cpu", weights_only=True),
                               camera_embedding=args.camera_embedding)
            except Exception as error:
                if not args.overwrite:
                    raise ValueError(f"Incomplete/invalid cache; use --overwrite to rebuild: {path}") from error
            else:
                if not args.overwrite:
                    print(f"Skip existing {path}")
                    return False
            path.unlink()
        if args.overwrite:
            (partial_dir / f"{sample['sample_id']}.pt").unlink(missing_ok=True)
        return True

    samples_to_encode = [sample for sample in samples if needs_encoding(sample)]
    if not samples_to_encode:
        print("Feature caches already complete")
        return

    # Persist identity before starting long encoder jobs. A crash must not leave
    # anonymous .partial tensors that a later, different configuration can reuse.
    temporary_metadata = output_dir / ".cache_manifest.tmp"
    cache_metadata = {
        "samples": sample_ids, "settings": cache_marker,
        "sample_signatures": sample_signatures,
    }
    temporary_metadata.write_text(json.dumps(cache_metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    partial_dir.mkdir(exist_ok=True)

    print("Loading frozen text encoder")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.base_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    image_processor = CLIPImageProcessor.from_pretrained(args.base_model_id, subfolder="image_processor")
    scheduler = UniPCMultistepScheduler.from_pretrained(args.base_model_id, subfolder="scheduler")
    transformer_config = WanTransformer3DModel.load_config(args.base_model_id, subfolder="transformer")
    if transformer_config.get("in_channels") != 36 or transformer_config.get("out_channels") != 16:
        raise ValueError("Static training requires the Wan I2V transformer (in=36, out=16), not T2V")
    # Real components provide the same registry/scale-factor behavior as normal
    # inference. This lightweight config object is used only for patch sizing.
    transformer_stub = type("TransformerConfig", (), {"config": transformer_config})()
    # Encoder-only pipeline; it is never saved or used for denoising.
    pipe = PCDControllerPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, image_encoder=None,
        image_processor=image_processor, transformer=transformer_stub,
        vae=None, scheduler=scheduler,
    )
    negative_cache = None
    for index, sample in enumerate(samples_to_encode):
        partial_path = partial_dir / f"{sample['sample_id']}.pt"
        if not partial_path.exists():
            with torch.no_grad():
                prompt_embeds = pipe._get_t5_prompt_embeds(sample["prompt"], device=device)
                if negative_cache is None:
                    negative_cache = pipe._get_t5_prompt_embeds(args.negative_prompt, device=device).cpu()
            cache = {"prompt_embeds": prompt_embeds[0].to(dtype).cpu(),
                     "negative_prompt_embeds": negative_cache[0].to(dtype)}
            temporary_path = partial_dir / f".{sample['sample_id']}.text.tmp"
            torch.save(cache, temporary_path)
            temporary_path.replace(partial_path)
        print(f"[{index + 1}/{len(samples)}] text cached: {sample['sample_id']}")
    del pipe, text_encoder
    gc.collect()
    empty_cache(device)

    print("Loading frozen CLIP image encoder")
    image_encoder = CLIPVisionModel.from_pretrained(
        args.base_model_id, subfolder="image_encoder", torch_dtype=torch.float32,
    ).to(device).eval()
    image_processor = CLIPImageProcessor.from_pretrained(args.base_model_id, subfolder="image_processor")
    for sample in samples_to_encode:
        partial_path = partial_dir / f"{sample['sample_id']}.pt"
        cache = torch.load(partial_path, map_location="cpu", weights_only=True)
        if "image_embeds" not in cache:
            with Image.open(sample["reference_image"]) as image:
                inputs = image_processor(images=image.convert("RGB"), return_tensors="pt").to(device)
                image_embeds = image_encoder(**inputs, output_hidden_states=True).hidden_states[-2]
            cache["image_embeds"] = image_embeds[0].to(dtype).cpu()
            temporary_path = partial_dir / f".{sample['sample_id']}.clip.tmp"
            torch.save(cache, temporary_path)
            temporary_path.replace(partial_path)
    del image_encoder
    gc.collect()
    empty_cache(device)

    print("Loading frozen Wan VAE")
    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_id, subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()
    # Reuse the real transformer config for spatial patch sizing; do not load
    # the 14B weights merely to construct an encoder-only pipeline.
    pipe = PCDControllerPipeline(
        tokenizer=None, text_encoder=None, image_encoder=None, image_processor=image_processor,
        transformer=transformer_stub, vae=vae, scheduler=scheduler,
    )
    for index, sample in enumerate(samples_to_encode):
        output_path = output_dir / f"{sample['sample_id']}.pt"
        cache = torch.load(partial_dir / f"{sample['sample_id']}.pt",
                           map_location="cpu", weights_only=True)
        required = {"prompt_embeds", "negative_prompt_embeds", "image_embeds"}
        if not required <= cache.keys():
            raise RuntimeError("Text/CLIP partial cache is incomplete; rerun the cache command")
        image, render_video, render_mask, camera_embedding, height, width = load_dataset(
            reference_image=sample["reference_image"], render_path=sample["render_path"],
            nframe=args.nframe, max_area=args.max_area, pipe=pipe,
            use_camera_embedding=args.camera_embedding, device=device,
        )
        target_video = video_tensor(sample["target_video"], height, width, args.nframe).to(device)
        render_video = render_video.to(device=device, dtype=torch.float32)
        target_latent = normalize_vae_latent(vae.encode(target_video).latent_dist.mode(), vae)
        render_latent = normalize_vae_latent(vae.encode(render_video).latent_dist.mode(), vae)
        preprocessed_image = pipe.video_processor.preprocess(image, height=height, width=width).to(device)
        _, condition = pipe.prepare_latents(
            preprocessed_image, batch_size=1, num_channels_latents=vae.config.z_dim,
            height=height, width=width, num_frames=args.nframe,
            dtype=torch.float32, device=device,
            latents=torch.zeros(1, vae.config.z_dim, (args.nframe - 1) // 4 + 1,
                                height // 8, width // 8, device=device),
        )
        # Ray construction remains on the selected device. Host transfer occurs
        # only when assembling the serialized cache, alongside the other tensors.
        camera_embedding = camera_embedding.float() if camera_embedding is not None else None
        cache.update({
            "target_latent": target_latent[0].to(dtype).cpu(),
            "condition": condition[0].to(dtype).cpu(),
            "render_latent": render_latent[0].to(dtype).cpu(),
            "render_mask": render_mask[0].float().cpu(),
            "camera_embedding": camera_embedding[0].to(dtype).cpu() if camera_embedding is not None else None,
            "height": height, "width": width, "nframe": args.nframe,
            "base_model_id": args.base_model_id,
        })
        validate_cache(cache, camera_embedding=args.camera_embedding)
        temporary_path = output_dir / f".{sample['sample_id']}.tmp"
        torch.save(cache, temporary_path)
        temporary_path.replace(output_path)
        (partial_dir / f"{sample['sample_id']}.pt").unlink()
        print(f"[{index + 1}/{len(samples)}] saved {output_path} ({output_path.stat().st_size / 2**20:.1f} MiB)")
        del target_video, render_video, target_latent, render_latent, condition, cache
        gc.collect()
        empty_cache(device)

    temporary_metadata = output_dir / ".cache_manifest.tmp"
    temporary_metadata.write_text(json.dumps(cache_metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    print("Feature caching complete")


if __name__ == "__main__":
    main()
