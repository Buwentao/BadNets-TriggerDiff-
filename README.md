# BadNets Backdoor + Fine-tuning / Fine-pruning (CIFAR-10)

This project implements:

- BadNets attack: add a bottom-right square trigger and relabel to a target class during training.
- Metrics: Clean Accuracy (CA) and Attack Success Rate (ASR).
- Defenses:
  - Fine-tuning on a clean subset
  - Fine-pruning (mask low-activation feature dims) + fine-tuning

## Install

You already have `torch/torchvision` installed in this environment. If needed:

```bash
pip install numpy tqdm pandas pyyaml matplotlib
```

## Recommended commands

From `backdoor_badnets/`:

```bash
# 1) Train clean model
python src/train_clean.py --seed 42 --epochs 50

# 2) Train backdoored model
python src/train_backdoor.py --poison_rate 0.05 --trigger_size 5 --target_label 9 --seed 42 --epochs 50

# 3) Evaluate CA/ASR
python src/eval.py --ckpt results/runs/bd_p0.05_t9_s5_seed42.pt --trigger_size 5 --target_label 9

# 4) Defense: fine-tune
python src/defense_finetune.py --ckpt results/runs/bd_p0.05_t9_s5_seed42.pt --lr 1e-3 --epochs 10

# 5) Defense: fine-prune + fine-tune
python src/defense_fineprune.py --ckpt results/runs/bd_p0.05_t9_s5_seed42.pt --prune_ratio 0.2 --lr 1e-3 --epochs 10

# 6) Run the full experiment suite (tables + figures)
python src/run_all.py --seed 42 --epochs_clean 50 --epochs_backdoor 50 --epochs_defense 10
```

## Outputs

- Checkpoints: `results/runs/*.pt`
- Tables: `results/tables/*.csv`
- Figures: `results/figures/*.png`
