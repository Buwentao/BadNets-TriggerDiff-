from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from backdoor.metrics import attack_success_rate, clean_accuracy
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--clean_subset", type=int, default=10000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=9)
    args = p.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, map_location=device)

    model = make_resnet18(num_classes=10).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

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
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        scheduler.step()
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

    out_path = paths.runs / f"def_ft_from_{ckpt_path.stem}.pt"
    save_checkpoint(
        out_path,
        {
            "kind": "defense_finetune",
            "seed": args.seed,
            "from_ckpt": str(ckpt_path),
            "epochs": args.epochs,
            "lr": args.lr,
            "clean_subset": args.clean_subset,
            "state_dict": model.state_dict(),
        },
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
