#!/bin/bash
# Test-set evaluation, paired: released V2 vs the seed-0 fine-tunes (plan 05), mdCATH + ATLAS.
#   sbatch investigation/slurm/eval_checkpoints.sbatch        (from the investigation worktree)
# L40S: the GPU the training cache was written on. Results: investigation/results/17_eval_*.json
#SBATCH -J eval_ckpt
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/eval_ckpt_%j.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts
nvidia-smi --query-gpu=name --format=csv,noheader; git log --oneline -1
R=$HOME/scratch/esmdyn_retrain/runs
models=(--model v2-released=V2)
for r in v2ft-dcm-only-fixed-s0 v2ft-all-heads-fixed-s0 v2ft-all-heads-bug-s0; do models+=(--model $r=$R/$r/best.pt); done
rc=0
for ds in mdcath atlas; do
    python -u investigation/scripts/eval_checkpoints.py --dataset $ds "${models[@]}" \
        --out investigation/results/17_eval_checkpoints_$ds.json || rc=1
done
exit $rc
