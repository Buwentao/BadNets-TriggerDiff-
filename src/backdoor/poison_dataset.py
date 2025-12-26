from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


TriggerFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PoisonSpec:
    poison_rate: float
    target_label: int
    seed: int = 42


class PoisonedDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        poison_rate: float,
        target_label: int,
        trigger_fn: TriggerFn,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= poison_rate <= 1.0):
            raise ValueError("poison_rate must be in [0,1]")
        self.base = base
        self.target_label = int(target_label)
        self.trigger_fn = trigger_fn

        n = len(base)
        m = int(round(poison_rate * n))
        rng = np.random.RandomState(seed)
        self.poison_idx: Set[int] = set(rng.choice(n, m, replace=False).tolist())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if idx in self.poison_idx:
            x = self.trigger_fn(x)
            y = self.target_label
        return x, y


