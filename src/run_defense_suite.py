from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from backdoor.metrics import attack_success_rate, clean_accuracy
from backdoor.resnet_feature import ResNet18WithLayer4Mask, ResNet18WithMask
from utils import ensure_dirs, get_device, load_cifar10, load_checkpoint, make_loader, make_resnet18, set_seed


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_cmd(root: Path, cmd: List[str]) -> None:
    subprocess.run([sys.executable, str(root / "src" / cmd[0]), *cmd[1:]], cwd=str(root), check=True)


def eval_ckpt(
    ckpt_path: Path,
    device: torch.device,
    test_loader,
    target_label: int,
    trigger_size: int,
    trigger_value: float,
) -> tuple[float, float]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    backbone = make_resnet18(num_classes=10).to(device)
    backbone.load_state_dict(ckpt["state_dict"], strict=True)
    if "layer4_mask" in ckpt:
        model = ResNet18WithLayer4Mask(backbone, channel_mask=ckpt["layer4_mask"].to(device))
    elif "mask" in ckpt:
        model = ResNet18WithMask(backbone, mask=ckpt["mask"].to(device))
    else:
        model = backbone
    ca = clean_accuracy(model, test_loader, device)
    asr = attack_success_rate(
        model,
        test_loader,
        device=device,
        target_label=target_label,
        trigger_size=trigger_size,
        trigger_value=trigger_value,
    )
    return ca, asr


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="backdoored checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=9)
    p.add_argument("--clean_subset", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--prune_ratios", type=str, default="0.1,0.2,0.3")
    args = p.parse_args()

    paths = ensure_dirs()
    root = paths.root
    set_seed(args.seed)
    device = get_device(args.device)

    test_ds = load_cifar10(paths.data, train=False)
    test_loader = make_loader(test_ds, batch_size=max(128, args.batch_size), train=False, num_workers=args.num_workers)

    rows = []

    # Fine-tuning
    run_cmd(
        root,
        [
            "defense_finetune.py",
            "--ckpt",
            args.ckpt,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--num_workers",
            str(args.num_workers),
            "--clean_subset",
            str(args.clean_subset),
            "--trigger_size",
            str(args.trigger_size),
            "--trigger_value",
            str(args.trigger_value),
            "--target_label",
            str(args.target_label),
        ],
    )
    ft_ckpt = root / "results" / "runs" / f"def_ft_from_{Path(args.ckpt).stem}.pt"
    ca, asr = eval_ckpt(ft_ckpt, device, test_loader, args.target_label, args.trigger_size, args.trigger_value)
    rows.append({"Setting": "Defense: Fine-tune", "CA": ca, "ASR": asr, "ckpt": str(ft_ckpt).replace("\\", "/")})
    print(f"finetune CA={ca:.4f} ASR={asr:.4f} ckpt={ft_ckpt}")

    # Fine-prune + FT
    for pr in parse_ratios(args.prune_ratios):
        run_cmd(
            root,
            [
                "defense_fineprune.py",
                "--ckpt",
                args.ckpt,
                "--prune_ratio",
                str(pr),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--num_workers",
                str(args.num_workers),
                "--clean_subset",
                str(args.clean_subset),
                "--trigger_size",
                str(args.trigger_size),
                "--trigger_value",
                str(args.trigger_value),
                "--target_label",
                str(args.target_label),
            ],
        )
        fp_ckpt = root / "results" / "runs" / f"def_fp{pr:.2f}_from_{Path(args.ckpt).stem}.pt"
        ca, asr = eval_ckpt(fp_ckpt, device, test_loader, args.target_label, args.trigger_size, args.trigger_value)
        rows.append(
            {
                "Setting": f"Defense: Prune{int(pr*100)}%+FT",
                "CA": ca,
                "ASR": asr,
                "prune_ratio": pr,
                "ckpt": str(fp_ckpt).replace("\\", "/"),
            }
        )
        print(f"prune={pr:.2f} CA={ca:.4f} ASR={asr:.4f} ckpt={fp_ckpt}")

    df = pd.DataFrame(rows)
    out_csv = paths.tables / "e3_defense.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote: {out_csv}")

    df_prune = df[df["Setting"].str.contains("Prune", na=False)].sort_values("prune_ratio")
    if len(df_prune) > 0:
        fig_path = paths.figures / "fig2_prune_vs_ca_asr.png"
        plt.figure(figsize=(5, 4))
        plt.plot(df_prune["prune_ratio"], df_prune["CA"], marker="o", label="CA")
        plt.plot(df_prune["prune_ratio"], df_prune["ASR"], marker="o", label="ASR")
        plt.xlabel("prune_ratio")
        plt.ylabel("metric")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"wrote: {fig_path}")


if __name__ == "__main__":
    main()
