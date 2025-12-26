from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from backdoor.metrics import attack_success_rate, clean_accuracy
from backdoor.resnet_feature import ResNet18WithLayer4Mask, ResNet18WithMask
from utils import ensure_dirs, get_device, load_cifar10, load_checkpoint, make_loader, maybe_subset, make_resnet18, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--trigger_size", type=int, default=5)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--target_label", type=int, default=9)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset_test", type=int, default=None, help="debug: limit test samples")
    p.add_argument("--out_csv", type=str, default=None, help="optional: append a row to CSV")
    args = p.parse_args()

    paths = ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    backbone = make_resnet18(num_classes=10).to(device)
    backbone.load_state_dict(ckpt["state_dict"], strict=True)
    if "layer4_mask" in ckpt:
        model = ResNet18WithLayer4Mask(backbone, channel_mask=ckpt["layer4_mask"].to(device))
    elif "mask" in ckpt:
        model = ResNet18WithMask(backbone, mask=ckpt["mask"].to(device))
    else:
        model = backbone

    test_ds = load_cifar10(paths.data, train=False)
    test_ds = maybe_subset(test_ds, args.subset_test, seed=args.seed + 1)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, train=False, num_workers=args.num_workers)

    ca = clean_accuracy(model, test_loader, device)
    asr = attack_success_rate(
        model,
        test_loader,
        device=device,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
    )

    print(f"ckpt={ckpt_path}")
    print(f"CA={ca:.4f}")
    print(f"ASR={asr:.4f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ckpt": str(ckpt_path).replace("\\", "/"),
            "CA": ca,
            "ASR": asr,
            "trigger_size": args.trigger_size,
            "trigger_value": args.trigger_value,
            "target_label": args.target_label,
        }
        for k in ("kind", "seed", "epochs", "poison_rate", "prune_ratio", "clean_subset", "from_ckpt"):
            if k in ckpt and k not in row:
                row[k] = ckpt[k]
        if out_path.exists():
            df = pd.read_csv(out_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(out_path, index=False)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
