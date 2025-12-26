from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
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


def eval_ca_asr(
    ckpt_path: Path,
    test_loader,
    device: torch.device,
    target_label: int,
    trigger_size: int,
    trigger_value: float,
) -> tuple[float, float]:
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
    return ca, asr


def maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean_ckpt", type=str, default="results/runs/clean_seed42.pt")
    p.add_argument("--e1_csv", type=str, default="results/tables/e1_poison_rate.csv")
    p.add_argument("--e2_csv", type=str, default="results/tables/e2_trigger_size.csv")
    p.add_argument("--e3_csv", type=str, default="results/tables/e3_defense.csv")
    p.add_argument("--target_label", type=int, default=9)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    test_ds = load_cifar10(paths.data, train=False)
    test_loader = make_loader(test_ds, batch_size=256, train=False, num_workers=0)

    rows = []

    clean_ckpt = paths.root / args.clean_ckpt
    if clean_ckpt.exists():
        ca, asr = eval_ca_asr(
            clean_ckpt, test_loader, device, args.target_label, args.trigger_size, args.trigger_value
        )
        rows.append({"Setting": "Clean model (p=0)", "CA": ca, "ASR": asr})

    e1 = maybe_read_csv(paths.root / args.e1_csv)
    if e1 is not None:
        for pr, name in [(0.01, "Backdoor p=1%"), (0.03, "Backdoor p=3%"), (0.05, "Backdoor p=5%"), (0.10, "Backdoor p=10%")]:
            hit = e1[np.isclose(e1["poison_rate"], pr)]
            if len(hit) > 0:
                r = hit.iloc[0]
                rows.append({"Setting": name, "CA": float(r["CA"]), "ASR": float(r["ASR"])})

    e2 = maybe_read_csv(paths.root / args.e2_csv)
    if e2 is not None:
        for ts in [3, 5]:
            hit = e2[e2["trigger_size"] == ts]
            if len(hit) > 0:
                r = hit.iloc[0]
                rows.append({"Setting": f"Backdoor p=5%, s={ts}", "CA": float(r["CA"]), "ASR": float(r["ASR"])})

    e3 = maybe_read_csv(paths.root / args.e3_csv)
    if e3 is not None:
        for _, r in e3.iterrows():
            rows.append({"Setting": r["Setting"], "CA": float(r["CA"]), "ASR": float(r["ASR"])})

    out = pd.DataFrame(rows)
    out_path = paths.tables / "main.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
