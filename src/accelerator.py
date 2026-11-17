"""Small CUDA/Ascend-NPU compatibility layer used by training and inference."""
import os
import random

import numpy as np
import torch


def npu_available():
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401  Registers torch.npu.
        except ImportError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def setup_device(name="auto", local_rank=None):
    """Select one device without importing CUDA-only launch dependencies."""
    local_rank = int(os.getenv("LOCAL_RANK", 0)) if local_rank is None else local_rank
    if name == "auto":
        name = "npu" if npu_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "npu":
        # Auto-registration exists in newer releases, but explicit import is
        # recommended by Ascend and makes the selected backend unambiguous.
        import torch_npu  # noqa: F401
        if not npu_available():
            raise RuntimeError("NPU requested but torch_npu/torch.npu is unavailable. Install CANN and the matching Ascend PyTorch wheel.")
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be auto, npu, cuda or cpu")
    return device


def device_module(device):
    if isinstance(device, str):
        device = torch.device(device)
    return torch.get_device_module(device.type)


def empty_cache(device):
    module = device_module(device)
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def synchronize(device):
    module = device_module(device)
    if hasattr(module, "synchronize"):
        module.synchronize()


def distributed_backend(device):
    return {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[torch.device(device).type]


def noautocast(device):
    """Disable autocast for numeric geometry without naming a CUDA device."""
    device_type = torch.device(device).type if not isinstance(device, torch.device) else device.type
    return torch.autocast(device_type=device_type, enabled=False)


def init_distributed(device):
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size <= 1:
        return False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=distributed_backend(device), init_method="env://",
            rank=int(os.getenv("RANK", 0)), world_size=world_size,
        )
    return True


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    module = device_module(device)
    if hasattr(module, "manual_seed_all"):
        module.manual_seed_all(seed)


def autocast_for(tensor, dtype=None, enabled=True):
    """Use the input's actual device; do not force a CUDA autocast on NPU."""
    return torch.autocast(device_type=tensor.device.type, dtype=dtype, enabled=enabled)


def sequence_parallel_world_size():
    """Avoid importing xFuser/flash-attn in single-device NPU jobs."""
    if int(os.getenv("WORLD_SIZE", 1)) == 1:
        return 1
    from xfuser.core.distributed import get_sequence_parallel_world_size
    return get_sequence_parallel_world_size()
