from __future__ import annotations

import torch


def apply_trigger(x: torch.Tensor, size: int = 5, value: float = 1.0) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected [C,H,W] tensor, got shape={tuple(x.shape)}")
    x = x.clone()
    _, h, w = x.shape
    if size <= 0 or size > min(h, w):
        raise ValueError(f"invalid size={size} for H={h}, W={w}")
    x[:, h - size : h, w - size : w] = float(value)
    return x


def apply_trigger_batch(x: torch.Tensor, size: int = 5, value: float = 1.0) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected [N,C,H,W] tensor, got shape={tuple(x.shape)}")
    x = x.clone()
    _, _, h, w = x.shape
    if size <= 0 or size > min(h, w):
        raise ValueError(f"invalid size={size} for H={h}, W={w}")
    x[:, :, h - size : h, w - size : w] = float(value)
    return x


