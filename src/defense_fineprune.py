from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm
from torchvision import datasets, transforms

from backdoor.metrics import attack_success_rate, clean_accuracy
from backdoor.resnet_feature import ResNet18WithLayer4Mask, extract_resnet18_layer4
from backdoor.trigger import apply_trigger_batch
from utils import (
    AverageMeter,
    cifar10_normalize,
    ensure_dirs,
    get_device,
    load_cifar10,
    load_checkpoint,
    make_loader,
    make_resnet18,
    maybe_subset,
    save_checkpoint,
    set_seed,
)


@torch.no_grad()
def compute_layer4_channel_mean_abs(
    backbone: nn.Module,
    loader,
    device: torch.device,
    apply_trigger: bool,
    trigger_size: int,
    trigger_value: float,
) -> torch.Tensor:
    backbone.eval()
    total = 0
    sum_abs = None
    for x, _ in tqdm(loader, desc="act-mean", leave=False):
        x = x.to(device)
        if apply_trigger:
            x = apply_trigger_batch(x, size=trigger_size, value=trigger_value)
        x = cifar10_normalize(x)
        h = extract_resnet18_layer4(backbone, x)  # (N,512,H,W)
        batch_sum = h.abs().sum(dim=(0, 2, 3))  # [512]
        if sum_abs is None:
            sum_abs = batch_sum
        else:
            sum_abs = sum_abs + batch_sum
        total += h.shape[0]
    if sum_abs is None:
        raise RuntimeError("empty loader")
    return sum_abs / max(1, total)


def build_mask_low_clean(act_mean: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in [0,1)")
    d = act_mean.numel()
    k = int(round(prune_ratio * d))
    if k <= 0:
        return torch.ones(d, dtype=torch.float32, device=act_mean.device)
    idx = torch.argsort(act_mean)[:k]
    mask = torch.ones(d, dtype=torch.float32, device=act_mean.device)
    mask[idx] = 0.0
    return mask


def build_mask_trigger_diff(clean_mean: torch.Tensor, trig_mean: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    if clean_mean.shape != trig_mean.shape:
        raise ValueError("clean_mean and trig_mean must have same shape")
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in [0,1)")
    d = clean_mean.numel()
    k = int(round(prune_ratio * d))
    if k <= 0:
        return torch.ones(d, dtype=torch.float32, device=clean_mean.device)

    score = trig_mean - clean_mean
    idx = torch.argsort(score, descending=True)[:k]
    mask = torch.ones(d, dtype=torch.float32, device=clean_mean.device)
    mask[idx] = 0.0
    return mask


def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer,
    criterion,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    for x, y in tqdm(loader, desc="finetune", leave=False):
        x = x.to(device)
        x = cifar10_normalize(x)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), n=y.numel())
    return loss_meter.avg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prune_ratio", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--clean_subset", type=int, default=10000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=9)
    p.add_argument(
        "--mask_mode",
        type=str,
        default="trigger_diff",
        choices=["trigger_diff", "low_clean"],
        help="How to choose pruned channels",
    )
    p.add_argument(
        "--mask_with_aug",
        action="store_true",
        help="Compute activation statistics using training augmentations (default: no aug).",
    )
    args = p.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, map_location=device)

    backbone = make_resnet18(num_classes=10).to(device)
    backbone.load_state_dict(ckpt["state_dict"], strict=True)

    if args.mask_with_aug:
        mask_ds = load_cifar10(paths.data, train=True)
        mask_ds = maybe_subset(mask_ds, args.clean_subset, seed=args.seed + 123)
        mask_loader = make_loader(mask_ds, batch_size=args.batch_size, train=False, num_workers=args.num_workers)
    else:
        mask_ds = datasets.CIFAR10(
            root=str(paths.data),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mask_ds = maybe_subset(mask_ds, args.clean_subset, seed=args.seed + 123)
        mask_loader = make_loader(mask_ds, batch_size=args.batch_size, train=False, num_workers=args.num_workers)

    clean_mean = compute_layer4_channel_mean_abs(
        backbone,
        mask_loader,
        device,
        apply_trigger=False,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
    )
    if args.mask_mode == "low_clean":
        mask = build_mask_low_clean(clean_mean, prune_ratio=args.prune_ratio)
    else:
        trig_mean = compute_layer4_channel_mean_abs(
            backbone,
            mask_loader,
            device,
            apply_trigger=True,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
        )
        mask = build_mask_trigger_diff(clean_mean, trig_mean, prune_ratio=args.prune_ratio)
    model = ResNet18WithLayer4Mask(backbone, channel_mask=mask).to(device)

    clean_train = load_cifar10(paths.data, train=True)
    clean_train = maybe_subset(clean_train, args.clean_subset, seed=args.seed + 123)
    train_loader = make_loader(clean_train, batch_size=args.batch_size, train=True, num_workers=args.num_workers)

    test_ds = load_cifar10(paths.data, train=False)
    test_loader = make_loader(test_ds, batch_size=max(128, args.batch_size), train=False, num_workers=args.num_workers)

    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        ca = clean_accuracy(model, test_loader, device)
        asr = attack_success_rate(
            model,
            test_loader,
            device=device,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
        )
        print(f"epoch={epoch} loss={loss:.4f} CA={ca:.4f} ASR={asr:.4f}")

    out_path = paths.runs / f"def_fp{args.prune_ratio:.2f}_from_{ckpt_path.stem}.pt"
    save_checkpoint(
        out_path,
        {
            "kind": "defense_fineprune",
            "seed": args.seed,
            "from_ckpt": str(ckpt_path),
            "prune_ratio": args.prune_ratio,
            "epochs": args.epochs,
            "lr": args.lr,
            "clean_subset": args.clean_subset,
            "layer4_mask": mask.detach().cpu(),
            "state_dict": backbone.state_dict(),
        },
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
