from __future__ import annotations

import argparse

import torch
from torchvision.utils import make_grid, save_image

from backdoor.trigger import apply_trigger
from utils import ensure_dirs, load_cifar10, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--index", type=int, default=0, help="which test image to visualize")
    args = p.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)

    ds = load_cifar10(paths.data, train=False)
    x, _ = ds[args.index]
    x2 = apply_trigger(x, size=args.trigger_size, value=args.trigger_value)

    grid = make_grid(torch.stack([x, x2], dim=0), nrow=2, padding=10)
    out = paths.figures / "trigger_example.png"
    save_image(grid, out)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()

