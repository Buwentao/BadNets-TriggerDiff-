from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from backdoor.trigger import apply_trigger_batch
from utils import cifar10_normalize


@torch.no_grad()
def clean_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        x = cifar10_normalize(x)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def attack_success_rate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_label: int,
    trigger_size: int = 5,
    trigger_value: float = 1.0,
) -> float:
    model.eval()
    target_label = int(target_label)

    success = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mask = y != target_label
        if mask.sum().item() == 0:
            continue
        x = x[mask]
        x = apply_trigger_batch(x, size=trigger_size, value=trigger_value)
        x = cifar10_normalize(x)
        logits = model(x)
        pred = logits.argmax(dim=1)
        success += (pred == target_label).sum().item()
        total += pred.numel()
    return success / max(1, total)
