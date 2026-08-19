"""GPU-oriented training helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def setup_gpu_training() -> torch.device:
    """Enable cuDNN autotune and return CUDA device."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def amp_setup(device: torch.device, enabled: bool) -> Tuple[bool, Optional[torch.dtype], bool]:
    """
    Return (use_amp, autocast_dtype, use_grad_scaler).

    Prefer bf16 on modern NVIDIA GPUs (Blackwell/Ada/Hopper): wider dynamic
    range than fp16, no GradScaler needed, far fewer Inf/NaN spikes.
    """
    if not enabled or device.type != "cuda":
        return False, None, False
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def maybe_compile(model: torch.nn.Module, enabled: bool = True) -> torch.nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception as exc:
        print(f"torch.compile disabled: {exc}", flush=True)
        return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile / DataParallel wrappers for isinstance checks."""
    return getattr(model, "_orig_mod", model)


def make_valid_mask(defended_len: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(B, seq_len) mask where positions < defended_len are 1."""
    positions = torch.arange(seq_len, device=defended_len.device)
    return (positions.unsqueeze(0) < defended_len.unsqueeze(1)).to(defended_len.dtype)


def move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.device != device:
                out[key] = value.to(device, non_blocking=non_blocking)
            else:
                out[key] = value
        else:
            out[key] = value
    return out
