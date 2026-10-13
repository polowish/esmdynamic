#!/bin/bash
# Plan 05 step 3: tests, a small window cache, timing, and a smoke run of option 1, option 2 and
# its control.   sbatch investigation/slurm/retrain_timing.sbatch   (from the investigation worktree)
#SBATCH -J retrain_timing
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/retrain_timing_%j.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail   # after activation (conda's cuda-nvcc activate script reads unset variables); no -e: every section reports
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts:$HOME/scratch/pylibs/wandb
C=$HOME/scratch/esmdyn_cache/timing
R=$HOME/scratch/esmdyn_retrain/smoke
rm -rf $R; mkdir -p $C $R        # caches are kept: deterministic, and existing shards are skipped
export WANDB_DIR=$R/wandb          # smoke runs: kept apart from the real runs
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1

echo "== tests (CPU parts)"
python investigation/tests/test_retrain.py 2>&1 | grep -E "^(PASS|FAIL|SKIP)|passed|Error"

echo "== small caches (eval-mode ESMFold)"
for args in "train all 96 0 mdcath_train" "val fixed 24 0 mdcath_val" "val fixed 16 256 long256"; do set -- $args
  python investigation/scripts/cache_esmfold_features.py --split $1 --windows $2 --esmfold-mode eval \
      --out $C/$5 --max-windows $3 --min-length $4 --plan-only | tail -1
  python investigation/scripts/cache_esmfold_features.py --split $1 --windows $2 --esmfold-mode eval \
      --out $C/$5 --max-windows $3 --min-length $4 | tail -1
done

echo "== cached == live"
RETRAIN_TEST_CACHE=$C/mdcath_train python investigation/tests/test_retrain.py cached_equals_live 2>&1 | tail -3

echo "== timing"
python investigation/scripts/retrain_timing.py --train-cache $C/mdcath_train --val-cache $C/mdcath_val \
    --long-cache $C/long256 --out $R/timing.json 2>&1 | grep -v Warning

echo "== smoke: option 1, option 2, control (2 epochs of 128 draws, 16 val windows)"
common="--seed 0 --train-cache $C/mdcath_train --val-cache $C/mdcath_val --n-train 128 --n-val 16 \
        --workers 2 --wandb-project esmdynamic-retrain-smoke --artifact-every 1"
python investigation/scripts/train_from_v2.py --trainable at-init --arm fixed $common --out $R/opt1_fixed --max-epochs 2 2>&1 | tail -2
python investigation/scripts/train_from_v2.py --trainable all --arm fixed $common --out $R/opt2_fixed --max-epochs 2 2>&1 | tail -2
python investigation/scripts/train_from_v2.py --trainable all --arm bug $common --out $R/opt2_bug --max-epochs 2 2>&1 | tail -2
echo "== resume: option 1 to epoch 3"
python investigation/scripts/train_from_v2.py --trainable at-init --arm fixed $common --out $R/opt1_fixed --max-epochs 3 --resume 2>&1 | tail -2
echo "== draws identical across runs?"
for e in 001 002; do cmp $R/opt1_fixed/plans/epoch_$e.npy $R/opt2_fixed/plans/epoch_$e.npy && \
    cmp $R/opt2_fixed/plans/epoch_$e.npy $R/opt2_bug/plans/epoch_$e.npy && echo "epoch $e identical"; done
python - <<EOF
import json
for r in ("opt1_fixed", "opt2_fixed", "opt2_bug"):
    h = json.load(open("$R/" + r + "/history.json"))
    print(r, [(x["epoch"], round(x["val_loss"], 2), round(x["val_bal_acc_320K"], 4),
               {k[11:]: round(v, 3) for k, v in x.items() if k.startswith("block_norm_")} or x.get("block_norms"))
              for x in h])
EOF
ls $R/wandb/wandb 2>/dev/null | head -5
cp $R/timing.json investigation/results/15_timing.json
echo DONE
