#!/bin/bash
# Plan 05 step 3: ESMFold window cache for mdCATH, eval mode.
#   sbatch --array=0-7 investigation/slurm/cache_mdcath.sbatch train all 8     # 35,012 windows, 8 tasks
#   sbatch --array=0   investigation/slurm/cache_mdcath.sbatch val fixed 1     # 270 windows
#   optional 4th argument: output dir (default ~/scratch/esmdyn_cache/mdcath_<split>), e.g. to
#   redo some tasks' shards beside the live cache before swapping them in
# (after the CPU plan step: cache_esmfold_features.py ... --plan-only). Resumable: existing shards are skipped.
#SBATCH -J cache_mdcath
#SBATCH -p gpu
# L40S only (user, 2026-10-15): ESMFold in bf16 is bit-reproducible on one architecture but
# not across them -- windows written on RTX PRO 6000 (Blackwell, sm_120) differed from an L40S
# recompute by up to 27% in s_z (investigation/scripts/verify_cache.py). One cache, one GPU type.
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 05:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/cache_mdcath_%A_%a.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts
SPLIT=$1; WINDOWS=$2; NTASKS=$3; OUT=${4:-$HOME/scratch/esmdyn_cache/mdcath_$SPLIT}
nvidia-smi --query-gpu=name --format=csv,noheader; git log --oneline -1
python investigation/scripts/cache_esmfold_features.py --split $SPLIT --windows $WINDOWS --esmfold-mode eval \
    --seed 0 --out $OUT --task $SLURM_ARRAY_TASK_ID --n-tasks $NTASKS
