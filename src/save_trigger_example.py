from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from backdoor.trigger import apply_trigger
from utils import ensure_dirs, load_cifar10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    paths = ensure_dirs()
    out_path = paths.figures / "trigger_example.png" if args.out is None else args.out

    ds = load_cifar10(paths.data, train=False)
    x, _ = ds[0]
    x_trig = apply_trigger(x, size=args.trigger_size, value=args.trigger_value)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(x.permute(1, 2, 0).numpy())
    axs[0].set_title("clean")
    axs[0].axis("off")
    axs[1].imshow(x_trig.permute(1, 2, 0).numpy())
    axs[1].set_title("triggered")
    axs[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

