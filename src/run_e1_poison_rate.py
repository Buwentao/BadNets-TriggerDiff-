from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from backdoor.metrics import attack_success_rate, clean_accuracy
from utils import ensure_dirs, get_device, load_cifar10, load_checkpoint, make_loader, make_resnet18, set_seed


def parse_rates(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def train_if_needed(
    root: Path,
    poison_rate: float,
    trigger_size: int,
    trigger_value: float,
    target_label: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    device: str | None,
    subset_train: int | None,
    subset_test: int | None,
    skip_existing: bool,
) -> Path:
    ckpt = root / "results" / "runs" / f"bd_p{poison_rate:.2f}_t{target_label}_s{trigger_size}_seed{seed}.pt"
    if skip_existing and ckpt.exists():
        return ckpt

    cmd = [
        sys.executable,
        str(root / "src" / "train_backdoor.py"),
        "--poison_rate",
        str(poison_rate),
        "--trigger_size",
        str(trigger_size),
        "--trigger_value",
        str(trigger_value),
        "--target_label",
        str(target_label),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--num_workers",
        str(num_workers),
    ]
    if device:
        cmd += ["--device", device]
    if subset_train:
        cmd += ["--subset_train", str(subset_train)]
    if subset_test:
        cmd += ["--subset_test", str(subset_test)]
    subprocess.run(cmd, cwd=str(root), check=True)
    return ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--poison_rates", type=str, default="0.01,0.03,0.05,0.10")
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--subset_train", type=int, default=None, help="debug: limit train samples")
    p.add_argument("--subset_test", type=int, default=None, help="debug: limit test samples")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    paths = ensure_dirs()
    root = paths.root
    set_seed(args.seed)
    device = get_device(args.device)

    test_ds = load_cifar10(paths.data, train=False)
    test_loader = make_loader(test_ds, batch_size=max(128, args.batch_size), train=False, num_workers=args.num_workers)

    rows = []
    for pr in parse_rates(args.poison_rates):
        ckpt_path = train_if_needed(
            root=root,
            poison_rate=pr,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
            target_label=args.target_label,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=args.device,
            subset_train=args.subset_train,
            subset_test=args.subset_test,
            skip_existing=args.skip_existing,
        )

        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model = make_resnet18(num_classes=10).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        ca = clean_accuracy(model, test_loader, device)
        asr = attack_success_rate(
            model,
            test_loader,
            device=device,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
        )
        rows.append(
            {
                "poison_rate": pr,
                "trigger_size": args.trigger_size,
                "target": args.target_label,
                "CA": ca,
                "ASR": asr,
                "ckpt": str(ckpt_path).replace("\\", "/"),
            }
        )
        print(f"p={pr:.2f} CA={ca:.4f} ASR={asr:.4f} ckpt={ckpt_path}")

    df = pd.DataFrame(rows).sort_values("poison_rate")
    out_csv = paths.tables / "e1_poison_rate.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote: {out_csv}")

    fig_path = paths.figures / "fig1_p_vs_asr.png"
    plt.figure(figsize=(5, 4))
    plt.plot(df["poison_rate"], df["ASR"], marker="o")
    plt.xlabel("poison_rate")
    plt.ylabel("ASR")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"wrote: {fig_path}")


if __name__ == "__main__":
    main()

