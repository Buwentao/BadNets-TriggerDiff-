from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from backdoor.metrics import clean_accuracy
from utils import (
    AverageMeter,
    cifar10_normalize,
    ensure_dirs,
    get_device,
    load_cifar10,
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
    for x, y in tqdm(loader, desc="train", leave=False):
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--subset_train", type=int, default=None, help="debug: limit train samples")
    p.add_argument("--subset_test", type=int, default=None, help="debug: limit test samples")
    args = p.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    train_ds = load_cifar10(paths.data, train=True)
    test_ds = load_cifar10(paths.data, train=False)
    train_ds = maybe_subset(train_ds, args.subset_train, seed=args.seed)
    test_ds = maybe_subset(test_ds, args.subset_test, seed=args.seed + 1)

    train_loader = make_loader(train_ds, batch_size=args.batch_size, train=True, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, train=False, num_workers=args.num_workers)

    model = make_resnet18(num_classes=10).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        scheduler.step()
        ca = clean_accuracy(model, test_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} CA={ca:.4f}")

    ckpt_path = paths.runs / f"clean_seed{args.seed}.pt"
    save_checkpoint(
        ckpt_path,
        {
            "kind": "clean",
            "seed": args.seed,
            "epochs": args.epochs,
            "state_dict": model.state_dict(),
        },
    )
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()

