from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class Paths:
    root: Path
    data: Path
    results: Path
    runs: Path
    tables: Path
    figures: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    return Paths(
        root=root,
        data=root / "data",
        results=results,
        runs=results / "runs",
        tables=results / "tables",
        figures=results / "figures",
    )


def ensure_dirs() -> Paths:
    paths = get_paths()
    paths.data.mkdir(parents=True, exist_ok=True)
    paths.runs.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cifar10_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose([transforms.ToTensor()])


def cifar10_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=x.device, dtype=x.dtype)
    std = torch.tensor(CIFAR10_STD, device=x.device, dtype=x.dtype)
    if x.ndim == 3:
        return (x - mean[:, None, None]) / std[:, None, None]
    if x.ndim == 4:
        return (x - mean[None, :, None, None]) / std[None, :, None, None]
    raise ValueError(f"expected 3D/4D tensor, got shape={tuple(x.shape)}")


def load_cifar10(data_root: Union[str, Path], train: bool, download: bool = True) -> Dataset:
    return datasets.CIFAR10(
        root=str(Path(data_root)),
        train=train,
        download=download,
        transform=cifar10_transforms(train=train),
    )


def maybe_subset(dataset: Dataset, n: Optional[int], seed: int) -> Dataset:
    if n is None or n <= 0 or n >= len(dataset):
        return dataset
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=int(n), replace=False).tolist()
    return Subset(dataset, idx)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    train: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location)
