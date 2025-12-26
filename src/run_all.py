from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dirs


def run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def plot_poison_rate(in_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(in_csv)
    if "poison_rate" not in df.columns or df.empty:
        return
    df = df.sort_values("poison_rate")
    plt.figure(figsize=(5, 3.2))
    plt.plot(df["poison_rate"], df["ASR"], marker="o")
    plt.xlabel("poison_rate")
    plt.ylabel("ASR")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_prune_ratio(in_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(in_csv)
    if "prune_ratio" not in df.columns or df.empty:
        return
    df = df.sort_values("prune_ratio")
    plt.figure(figsize=(5, 3.2))
    plt.plot(df["prune_ratio"], df["CA"], marker="o", label="CA")
    plt.plot(df["prune_ratio"], df["ASR"], marker="o", label="ASR")
    plt.xlabel("prune_ratio")
    plt.ylabel("metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--epochs_clean", type=int, default=50)
    p.add_argument("--epochs_backdoor", type=int, default=50)
    p.add_argument("--epochs_defense", type=int, default=10)

    p.add_argument("--target_label", type=int, default=9)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--main_trigger_size", type=int, default=5)
    p.add_argument("--poison_rates", type=str, default="0.01,0.03,0.05,0.10")
    p.add_argument("--trigger_sizes", type=str, default="3,5")
    p.add_argument("--clean_subset", type=int, default=5000)
    p.add_argument("--prune_ratios", type=str, default="0.10,0.20,0.30")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    paths = ensure_dirs()
    py = sys.executable

    poison_rates = [float(x) for x in args.poison_rates.split(",") if x.strip()]
    trigger_sizes = [int(x) for x in args.trigger_sizes.split(",") if x.strip()]
    prune_ratios = [float(x) for x in args.prune_ratios.split(",") if x.strip()]

    main_csv = paths.tables / "main.csv"
    if main_csv.exists():
        main_csv.unlink()

    trained_ckpts: set[Path] = set()

    # 0) example figure
    run(
        [
            py,
            "src/save_trigger_example.py",
            "--trigger_size",
            str(args.main_trigger_size),
            "--trigger_value",
            str(args.trigger_value),
        ]
    )

    # 1) clean baseline
    clean_ckpt = paths.runs / f"clean_seed{args.seed}.pt"
    if not (args.skip_existing and clean_ckpt.exists()):
        run(
            [
                py,
                "src/train_clean.py",
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs_clean),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                *(["--device", args.device] if args.device else []),
            ]
        )
    trained_ckpts.add(clean_ckpt)
    run(
        [
            py,
            "src/eval.py",
            "--ckpt",
            str(clean_ckpt),
            "--trigger_size",
            str(args.main_trigger_size),
            "--trigger_value",
            str(args.trigger_value),
            "--target_label",
            str(args.target_label),
            "--out_csv",
            str(main_csv),
            *(["--device", args.device] if args.device else []),
        ]
    )

    # 2) E1: poison rate sweep (fixed trigger size)
    e1_csv = paths.tables / "e1_poison_rate.csv"
    if e1_csv.exists():
        e1_csv.unlink()
    for pr in poison_rates:
        ckpt = paths.runs / f"bd_p{pr:.2f}_t{args.target_label}_s{args.main_trigger_size}_seed{args.seed}.pt"
        if ckpt not in trained_ckpts and not (args.skip_existing and ckpt.exists()):
            run(
                [
                    py,
                    "src/train_backdoor.py",
                    "--poison_rate",
                    str(pr),
                    "--trigger_size",
                    str(args.main_trigger_size),
                    "--trigger_value",
                    str(args.trigger_value),
                    "--target_label",
                    str(args.target_label),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs_backdoor),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    *(["--device", args.device] if args.device else []),
                ]
            )
        trained_ckpts.add(ckpt)
        for out_csv in (e1_csv, main_csv):
            run(
                [
                    py,
                    "src/eval.py",
                    "--ckpt",
                    str(ckpt),
                    "--trigger_size",
                    str(args.main_trigger_size),
                    "--trigger_value",
                    str(args.trigger_value),
                    "--target_label",
                    str(args.target_label),
                    "--out_csv",
                    str(out_csv),
                    *(["--device", args.device] if args.device else []),
                ]
            )

    plot_poison_rate(e1_csv, paths.figures / "fig1_p_vs_asr.png")

    # 3) E2: trigger size ablation (fixed poison rate=0.05)
    e2_csv = paths.tables / "e2_trigger_size.csv"
    if e2_csv.exists():
        e2_csv.unlink()
    pr_fixed = 0.05
    for ts in trigger_sizes:
        ckpt = paths.runs / f"bd_p{pr_fixed:.2f}_t{args.target_label}_s{ts}_seed{args.seed}.pt"
        if ckpt not in trained_ckpts and not (args.skip_existing and ckpt.exists()):
            run(
                [
                    py,
                    "src/train_backdoor.py",
                    "--poison_rate",
                    str(pr_fixed),
                    "--trigger_size",
                    str(ts),
                    "--trigger_value",
                    str(args.trigger_value),
                    "--target_label",
                    str(args.target_label),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs_backdoor),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    *(["--device", args.device] if args.device else []),
                ]
            )
        trained_ckpts.add(ckpt)
        for out_csv in (e2_csv, main_csv):
            run(
                [
                    py,
                    "src/eval.py",
                    "--ckpt",
                    str(ckpt),
                    "--trigger_size",
                    str(ts),
                    "--trigger_value",
                    str(args.trigger_value),
                    "--target_label",
                    str(args.target_label),
                    "--out_csv",
                    str(out_csv),
                    *(["--device", args.device] if args.device else []),
                ]
            )

    # 4) E3: defenses on the main setting (p=0.05, s=main_trigger_size)
    base_bd = paths.runs / f"bd_p0.05_t{args.target_label}_s{args.main_trigger_size}_seed{args.seed}.pt"
    if not base_bd.exists():
        print(f"Missing base backdoor checkpoint: {base_bd}")
        print("Run with --poison_rates including 0.05 (default) or train it first.")
        return

    # Fine-tuning
    def_ft = paths.runs / f"def_ft_from_{base_bd.stem}.pt"
    if def_ft not in trained_ckpts and not (args.skip_existing and def_ft.exists()):
        run(
            [
                py,
                "src/defense_finetune.py",
                "--ckpt",
                str(base_bd),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs_defense),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--clean_subset",
                str(args.clean_subset),
                "--trigger_size",
                str(args.main_trigger_size),
                "--trigger_value",
                str(args.trigger_value),
                "--target_label",
                str(args.target_label),
                *(["--device", args.device] if args.device else []),
            ]
        )
    trained_ckpts.add(def_ft)
    run(
        [
            py,
            "src/eval.py",
            "--ckpt",
            str(def_ft),
            "--trigger_size",
            str(args.main_trigger_size),
            "--trigger_value",
            str(args.trigger_value),
            "--target_label",
            str(args.target_label),
            "--out_csv",
            str(main_csv),
            *(["--device", args.device] if args.device else []),
        ]
    )

    # Fine-pruning sweep
    e3_csv = paths.tables / "e3_fineprune.csv"
    if e3_csv.exists():
        e3_csv.unlink()
    for rr in prune_ratios:
        out_fp = paths.runs / f"def_fp{rr:.2f}_from_{base_bd.stem}.pt"
        if out_fp not in trained_ckpts and not (args.skip_existing and out_fp.exists()):
            run(
                [
                    py,
                    "src/defense_fineprune.py",
                    "--ckpt",
                    str(base_bd),
                    "--prune_ratio",
                    str(rr),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs_defense),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    "--clean_subset",
                    str(args.clean_subset),
                    "--trigger_size",
                    str(args.main_trigger_size),
                    "--trigger_value",
                    str(args.trigger_value),
                    "--target_label",
                    str(args.target_label),
                    *(["--device", args.device] if args.device else []),
                ]
            )
        trained_ckpts.add(out_fp)
        for out_csv in (e3_csv, main_csv):
            run(
                [
                    py,
                    "src/eval.py",
                    "--ckpt",
                    str(out_fp),
                    "--trigger_size",
                    str(args.main_trigger_size),
                    "--trigger_value",
                    str(args.trigger_value),
                    "--target_label",
                    str(args.target_label),
                    "--out_csv",
                    str(out_csv),
                    *(["--device", args.device] if args.device else []),
                ]
            )

    plot_prune_ratio(e3_csv, paths.figures / "fig2_prune_vs_ca_asr.png")
    print("Done.")


if __name__ == "__main__":
    main()
