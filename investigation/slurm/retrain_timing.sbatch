#!/bin/bash
# Plan 04 step 3: tests, small caches, timing, and an end-to-end smoke run of both stages, both arms.
#   sbatch investigation/slurm/retrain_timing.sbatch      (from the investigation worktree)
#SBATCH -J retrain_timing
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/retrain_timing_%j.out
set -euo pipefail
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts
C=$HOME/scratch/esmdyn_cache/timing
R=$HOME/scratch/esmdyn_retrain/smoke
rm -rf $C $R; mkdir -p $C $R
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== CPU tests"; python -m pytest -q investigation/tests/test_train_faithful.py -k "not cached_equals_live" 2>&1 | tail -5

echo "== small caches"
python investigation/scripts/cache_esmfold_features.py --dataset mdcath --split train --crops-per-long 3 --seed 0 --out $C/mdcath_train --max-crops 64
python investigation/scripts/cache_esmfold_features.py --dataset mdcath --split val --crops-per-long 1 --seed 0 --out $C/mdcath_val --max-crops 64
python investigation/scripts/cache_esmfold_features.py --dataset rcsb --split train --crops-per-long 2 --seed 0 --out $C/rcsb_train --max-crops 64
python investigation/scripts/cache_esmfold_features.py --dataset rcsb --split val --crops-per-long 1 --seed 0 --out $C/rcsb_val --max-crops 64

echo "== test 3: cached == live (train-mode pool and eval-mode pool)"
RETRAIN_TEST_CACHE=$C/mdcath_train python -m pytest -q -s investigation/tests/test_train_faithful.py -k cached_equals_live 2>&1 | tail -4
RETRAIN_TEST_CACHE=$C/mdcath_val   python -m pytest -q -s investigation/tests/test_train_faithful.py -k cached_equals_live 2>&1 | tail -4

echo "== timing"
python investigation/scripts/retrain_timing.py --mdcath-cache $C/mdcath_train --rcsb-cache $C/rcsb_train --out $R/timing.json

echo "== smoke: both stages, both arms"
for A in A B; do
  python investigation/scripts/train_faithful.py --stage pretrain --arm $A --seed 0 --train-cache $C/rcsb_train --val-cache $C/rcsb_val \
     --out $R/${A}_pretrain --n-train 128 --n-val 16 --max-epochs 2 --workers 2 | tail -3
  python investigation/scripts/train_faithful.py --stage finetune --arm $A --seed 0 --train-cache $C/mdcath_train --val-cache $C/mdcath_val \
     --init $R/${A}_pretrain/best.pt --out $R/${A}_finetune --n-train 128 --n-val 16 --max-epochs 3 --aux-heads-after 1 --workers 2 | tail -4
done
echo "== plans identical across arms?"
for s in pretrain finetune; do diff -rq $R/A_$s/plans $R/B_$s/plans && echo "$s plans identical"; done
python - <<EOF
import json
for a in "AB":
    for s in ("pretrain", "finetune"):
        h = json.load(open(f"$R/{a}_{s}/history.json"))
        print(a, s, [(x["epoch"], x["phase"], round(x["val_loss"], 3), {k: round(v, 4) for k, v in x.items() if k.startswith("block_norm")}) for x in h])
EOF
cp $R/timing.json investigation/results/15_timing.json
echo DONE
