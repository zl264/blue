"""Static-scene LoRA training helpers shared by caching, training and inference."""
import json
import os
import shutil
from pathlib import Path

import torch


CACHE_KEYS = (
    "target_latent", "condition", "render_latent", "render_mask",
    "camera_embedding", "prompt_embeds", "negative_prompt_embeds", "image_embeds",
)
CACHE_TENSOR_DIMS = {
    "target_latent": 4, "condition": 4, "render_latent": 4,
    "render_mask": 4, "camera_embedding": 4,
    "prompt_embeds": 2, "negative_prompt_embeds": 2, "image_embeds": 2,
}


def validate_cache(cache, camera_embedding=True):
    """Validate one unbatched cache before it can be treated as complete."""
    missing = set(CACHE_KEYS) - cache.keys()
    if missing:
        raise ValueError(f"Cache misses required keys: {sorted(missing)}")
    for key, dimensions in CACHE_TENSOR_DIMS.items():
        value = cache[key]
        if key == "camera_embedding" and value is None:
            if camera_embedding:
                raise ValueError("This cache configuration requires camera_embedding")
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != dimensions:
            raise ValueError(f"Cached {key} must be an unbatched {dimensions}D tensor")
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
             for key, value in cache.items() if key in CACHE_KEYS}
    validate_cache_batch(batch)


def move_cache(cache, device):
    """Move a DataLoader batch without adding a second batch dimension."""
    return {key: value.to(device=device, non_blocking=False)
            if isinstance(value, torch.Tensor) else value for key, value in cache.items()}


def validate_cache_batch(batch):
    expected_dims = {
        "target_latent": 5, "condition": 5, "render_latent": 5,
        "render_mask": 5, "camera_embedding": 5,
        "prompt_embeds": 3, "negative_prompt_embeds": 3, "image_embeds": 3,
    }
    for key, dimensions in expected_dims.items():
        value = batch.get(key)
        if key == "camera_embedding" and value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != dimensions:
            raise ValueError(f"Cached {key} must be a {dimensions}D tensor, got {type(value).__name__}")
    target_shape = batch["target_latent"].shape
    batch_size, channels, frames, height, width = target_shape
    if channels != 16:
        raise ValueError("target_latent must have 16 Wan VAE channels")
    if batch["condition"].shape != (batch_size, 20, frames, height, width):
        raise ValueError("Wan I2V condition must be [B,20,F,H,W] (4 mask + 16 image latent)")
    if batch["render_latent"].shape != target_shape:
        raise ValueError("target_latent and render_latent shapes must match")
    if batch["render_mask"].shape[:2] != (batch_size, 1):
        raise ValueError("render_mask must have shape [B,1,F,H,W]")
    raw_height, raw_width = batch["render_mask"].shape[-2:]
    raw_frames = (frames - 1) * 4 + 1
    if batch["render_mask"].shape[2:] != (raw_frames, raw_height, raw_width):
        raise ValueError("render_mask temporal/spatial dimensions are inconsistent")
    camera = batch.get("camera_embedding")
    if camera is not None and camera.shape != (batch_size, 6, raw_frames, raw_height, raw_width):
        raise ValueError("camera_embedding must be [B,6,nframe,H,W] with six Plücker-ray channels")
    for key in ("prompt_embeds", "negative_prompt_embeds", "image_embeds"):
        if batch[key].shape[0] != batch_size:
            raise ValueError(f"{key} batch dimension does not match the latent batch")


def normalize_vae_latent(latent, vae):
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
    inverse_std = 1 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
    return (latent - mean) * inverse_std


def load_manifest(path):
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            required = {"sample_id", "reference_image", "render_path", "target_video", "prompt"}
            missing = required - sample.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} misses {sorted(missing)}")
            if (not sample["sample_id"] or sample["sample_id"] in {".", ".."}
                    or Path(sample["sample_id"]).name != sample["sample_id"]):
                raise ValueError(f"{path}:{line_number}: sample_id must be one safe filename component")
            for key in ("reference_image", "render_path", "target_video"):
                sample[key] = str((path.parent / sample[key]).resolve())
            samples.append(sample)
    if not samples:
        raise ValueError(f"No samples in {path}")
    return samples


def add_controlnet_lora(transformer, rank=16, alpha=None, target="controlnet"):
    """Freeze Wan and add LoRA to explicitly matched controller Linear layers.

    target=controlnet affects controller blocks. target=all also affects Wan
    self/cross-attention, requiring more activation memory.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    if target not in ("controlnet", "all"):
        raise ValueError("target must be controlnet or all")
    if not hasattr(PeftModel, "active_adapters"):
        raise RuntimeError("This training script requires a PEFT version with PeftModel.active_adapters")
    targets = []
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        # Keep the first static-scene run small: adapt the controller blocks,
        # not all 14B Wan projections or high-resolution 3D convolutions.
        is_control = name.startswith("controlnet.")
        if target == "controlnet" and not is_control:
            continue
        if target == "all":
            is_attention = name.endswith(("to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"))
            if not (is_control or is_attention):
                continue
        targets.append(name)
    if not targets:
        raise ValueError(f"No LoRA target Linear modules found for {target!r}")
    config = LoraConfig(
        r=rank, lora_alpha=alpha or rank, lora_dropout=0.0, bias="none",
        target_modules=targets, init_lora_weights=True,
    )
    # Check the match before constructing the adapter; PEFT otherwise accepts a
    # valid pattern that happens to match no module in a custom model subclass.
    transformer = get_peft_model(transformer, config)
    # get_peft_model freezes base weights. Explicitly reject any unfrozen base
    # parameter so a future PEFT change cannot silently turn this into full FT.
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad = "lora_" in name
    if not any(parameter.requires_grad for parameter in transformer.parameters()):
        raise RuntimeError("PEFT did not create any trainable LoRA parameters")
    transformer.print_trainable_parameters()
    return transformer


def _write_adapter(transformer, output_dir, metadata):
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    if isinstance(transformer, PeftModel):
        adapter_name = transformer.active_adapters[0]
    else:
        adapter_name = transformer.active_adapters()[0]
    state_dict = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    state_dict = {key: value.detach().cpu().contiguous()
                  for key, value in state_dict.items()}
    if not state_dict:
        raise RuntimeError("No LoRA tensors were selected for saving")
    # Strip PeftModel's outer prefix. Diffusers injects directly into the
    # PCDController, so saved keys must start at its actual child modules.
    state_dict = {key.removeprefix("base_model.model."): value
                  for key, value in state_dict.items()}
    transformer.peft_config[adapter_name].save_pretrained(str(output_dir))
    save_file(state_dict, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    if metadata is not None:
        with (output_dir / "training_info.json").open("w", encoding="utf-8") as writer:
            json.dump(metadata, writer, indent=2, ensure_ascii=False)


def save_lora(transformer, output_dir, metadata=None):
    """Save only adapter tensors; atomically replace a complete checkpoint."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_adapter(transformer, temporary, metadata)
        # Rename the old complete adapter aside until the replacement is ready;
        # a failed save therefore cannot masquerade as the latest new weights.
        previous = output_dir.with_name(f".{output_dir.name}.old-{os.getpid()}")
        if previous.exists():
            shutil.rmtree(previous)
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError(f"Adapter destination is not a directory: {output_dir}")
            output_dir.replace(previous)
        try:
            temporary.replace(output_dir)
        except Exception:
            if previous.exists():
                previous.replace(output_dir)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_lora(transformer, path, adapter_name="static_scene", trainable=False):
    """Inject and strictly verify an adapter without a PEFT outer wrapper.

    Keeping PCDController as the outer object is important for diffusers CPU
    offload and for Uni3C's controlnet/gradient-checkpointing attributes.
    """
    if not hasattr(transformer, "load_lora_adapter"):
        raise RuntimeError("PCDController must inherit diffusers PeftAdapterMixin")
    if not Path(path).is_dir():
        raise FileNotFoundError(f"LoRA directory does not exist: {path}")
    from peft.utils.constants import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
    from safetensors.torch import load_file

    weights_path = Path(path) / "adapter_model.safetensors"
    saved = load_file(str(weights_path), device="cpu")
    if not saved:
        raise ValueError(f"LoRA checkpoint contains no tensors: {weights_path}")
    transformer.load_lora_adapter(
        str(path), prefix=None, adapter_name=adapter_name, low_cpu_mem_usage=True,
    )
    transformer.set_adapter(adapter_name)
    try:
        tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[
            transformer.peft_config[adapter_name].peft_type
        ]
        mapped = _insert_adapter_name_into_state_dict(
            saved, adapter_name=adapter_name, parameter_prefix=tuner_prefix,
        )
        model_state = transformer.state_dict()
        expected = {key for key in model_state
                    if tuner_prefix in key and f".{adapter_name}." in key}
        if set(mapped) != expected:
            raise RuntimeError(
                f"LoRA key mismatch: unexpected={sorted(set(mapped) - expected)[:5]}, "
                f"missing={sorted(expected - set(mapped))[:5]}"
            )
        for saved_key, value in saved.items():
            actual = model_state[mapped[saved_key]].detach().cpu()
            if actual.shape != value.shape or not torch.equal(actual, value):
                raise RuntimeError(f"LoRA weights did not round-trip exactly: {saved_key}")
        for name, parameter in transformer.named_parameters():
            parameter.requires_grad = trainable and "lora_" in name
        if trainable and not any(parameter.requires_grad for parameter in transformer.parameters()):
            raise RuntimeError("No trainable LoRA parameters after loading")
        if not trainable:
            transformer.eval()
    except Exception:
        transformer.delete_adapters(adapter_name)
        raise
    return transformer
