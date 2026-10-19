#!/bin/bash
# Smoke test of the optimisation options + --min-sep on the real caches: 2 epochs, then a resume
# to 3; every new option at once. Checks the resume, the EMA checkpoint, and that epoch 0's
# val_loss_unmasked reproduces the baseline run's epoch-0 val loss (1338.32).
#   sbatch investigation/slurm/smoke_optim.sbatch     (from the investigation worktree)
#SBATCH -J smoke_optim
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/smoke_optim_%j.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts:$HOME/scratch/pylibs/wandb
C=$HOME/scratch/esmdyn_cache; R=$HOME/scratch/esmdyn_retrain/smoke/optim
rm -rf $R; mkdir -p $R
git log --oneline -1
common=(--trainable all --arm fixed --seed 0 --train-cache $C/mdcath_train --val-cache $C/mdcath_val
        --out $R --n-train 64 --batch 16 --lr-blocks 3e-4 --lr-other 3e-5 --schedule cosine
        --warmup-epochs 0.5 --cosine-epochs 3 --ema-decay 0.9 --min-sep 3 --wandb off)
python -u investigation/scripts/train_from_v2.py "${common[@]}" --max-epochs 2 2>&1 | grep -vE "Warning|autocast|^\s+with" | tail -4
python -u investigation/scripts/train_from_v2.py "${common[@]}" --max-epochs 3 --resume 2>&1 | grep -vE "Warning|autocast|^\s+with" | grep -E "resumed|cap raised|finished" 
python - <<PY
import json, torch
h = json.load(open("$R/history.json")); m = json.load(open("$R/run_meta.json"))
print("epochs:", [e["epoch"] for e in h], "| lr groups", m["lr_groups"], "| steps/epoch", m["steps_per_epoch"], "| warmup", m["warmup_steps_total"])
print("epoch 0: val_loss (masked) %.2f, val_loss_unmasked %.2f (baseline 1338.32), ema_val_loss %s" % (h[0]["val_loss"], h[0]["val_loss_unmasked"], h[0].get("ema_val_loss")))
for e in h[1:]:
    print("epoch %d: val %.2f unmasked %.2f ema %.2f ema_unmasked %.2f" % (e["epoch"], e["val_loss"], e["val_loss_unmasked"], e["ema_val_loss"], e["ema_val_loss_unmasked"]))
s = [json.loads(l) for l in open("$R/steps.jsonl")]
print("lr by step (blocks / other):", [(round(x["lr"], 7), round(x["lr_last_group"], 8)) for x in s])
b = torch.load("$R/best_ema.pt", map_location="cpu", weights_only=False); l = torch.load("$R/last.pt", map_location="cpu", weights_only=False)
print("best_ema epoch", b["epoch"], "| last.pt has ema:", "ema" in l, "| DONE:", open("$R/DONE").read())
PY
