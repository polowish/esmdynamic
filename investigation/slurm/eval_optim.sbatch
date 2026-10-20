#!/bin/bash
# Test-set evaluation of the Phase 1 optimisation runs, paired against the constant-lr baseline
# (plan 05, optimised option 2). Same scoring as eval_checkpoints.sbatch; results/18_eval_optim_*.json.
#   sbatch investigation/slurm/eval_optim.sbatch        (from the investigation worktree)
#SBATCH -J eval_optim
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/eval_optim_%j.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts
nvidia-smi --query-gpu=name --format=csv,noheader; git log --oneline -1
R=$HOME/scratch/esmdyn_retrain/runs
models=()       # the first is the reference for the paired differences
for r in v2ft-all-heads-fixed-s0 v2ft-all-heads-fixed-mask-cos-s0 v2ft-all-heads-fixed-mask-grouplr-cos-s0; do
    models+=(--model $r=$R/$r/best.pt)
done
models+=(--model v2-released=V2)
rc=0
for ds in mdcath atlas; do
    python -u investigation/scripts/eval_checkpoints.py --dataset $ds "${models[@]}" \
        --out investigation/results/18_eval_optim_$ds.json || rc=1
done
exit $rc
