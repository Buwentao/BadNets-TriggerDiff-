from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from backdoor.metrics import attack_success_rate, clean_accuracy
from backdoor.resnet_feature import ResNet18WithLayer4Mask, ResNet18WithMask
from utils import ensure_dirs, get_device, load_cifar10, load_checkpoint, make_loader, make_resnet18, set_seed


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    backbone = make_resnet18(num_classes=10).to(device)
    backbone.load_state_dict(ckpt["state_dict"], strict=True)

    if "layer4_mask" in ckpt:
        return ResNet18WithLayer4Mask(backbone, channel_mask=ckpt["layer4_mask"].to(device))
    if "mask" in ckpt:
        return ResNet18WithMask(backbone, mask=ckpt["mask"].to(device))
    return backbone


def eval_row(
    ckpt_path: Path,
    test_loader,
    device: torch.device,
    target_label: int,
    trigger_size: int,
    trigger_value: float,
) -> dict:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = load_model_from_ckpt(ckpt_path, device)
    ca = clean_accuracy(model, test_loader, device)
    asr = attack_success_rate(
        model,
        test_loader,
        device=device,
        target_label=target_label,
        trigger_size=trigger_size,
        trigger_value=trigger_value,
    )
    row = {
        "ckpt": str(ckpt_path).replace("\\", "/"),
        "CA": ca,
        "ASR": asr,
        "trigger_size": trigger_size,
        "trigger_value": trigger_value,
        "target_label": target_label,
    }
    for k in ("kind", "seed", "epochs", "poison_rate", "prune_ratio", "clean_subset", "from_ckpt"):
        if k in ckpt:
            row[k] = ckpt[k]
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=9)
    p.add_argument("--runs_dir", type=str, default="results/runs")
    p.add_argument("--out_csv", type=str, default="results/tables/e3_fineprune.csv")
    p.add_argument("--out_defense_csv", type=str, default="results/tables/e3_defense.csv")
    p.add_argument("--out_png", type=str, default="results/figures/fig2_prune_vs_ca_asr.png")
    args = p.parse_args()

    plt.switch_backend("Agg")
    paths = ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    test_ds = load_cifar10(paths.data, train=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, train=False, num_workers=args.num_workers)

    runs_dir = Path(args.runs_dir)
    ckpts = sorted(runs_dir.glob("def_fp*_from_*.pt"))
    if not ckpts:
        raise SystemExit(f"No fine-prune ckpts found under: {runs_dir}")

    rows = [
        eval_row(
            ckpt_path=ckpt,
            test_loader=test_loader,
            device=device,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
        )
        for ckpt in ckpts
    ]
    df = pd.DataFrame(rows)
    if "prune_ratio" in df.columns:
        df = df.sort_values("prune_ratio")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote: {out_csv}")

    # Build a small comparison table (fine-tune + prune-k% entries if present).
    defense_rows = []
    ft_ckpts = sorted(runs_dir.glob("def_ft_from_*.pt"))
    if ft_ckpts:
        r = eval_row(
            ckpt_path=ft_ckpts[-1],
            test_loader=test_loader,
            device=device,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
        )
        defense_rows.append(
            {"Setting": "Defense: Fine-tune", "CA": r["CA"], "ASR": r["ASR"], "ckpt": r["ckpt"], "prune_ratio": ""}
        )

    for _, r in df.iterrows():
        pr = float(r.get("prune_ratio", float("nan")))
        defense_rows.append(
            {
                "Setting": f"Defense: Prune{int(round(pr * 100))}%+FT",
                "CA": float(r["CA"]),
                "ASR": float(r["ASR"]),
                "ckpt": r["ckpt"],
                "prune_ratio": pr,
            }
        )

    out_def_csv = Path(args.out_defense_csv)
    out_def_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(defense_rows).to_csv(out_def_csv, index=False)
    print(f"wrote: {out_def_csv}")

    # Plot
    if "prune_ratio" in df.columns and len(df) > 0:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5, 3.4))
        plt.plot(df["prune_ratio"], df["CA"], marker="o", label="CA")
        plt.plot(df["prune_ratio"], df["ASR"], marker="o", label="ASR")
        plt.xlabel("prune_ratio")
        plt.ylabel("metric")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()

