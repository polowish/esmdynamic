#!/bin/bash
# Plan 05 step 3: ESMFold window cache for mdCATH, eval mode.
#   sbatch --array=0-7 investigation/slurm/cache_mdcath.sbatch train all 8     # 35,012 windows, 8 tasks
#   sbatch --array=0   investigation/slurm/cache_mdcath.sbatch val fixed 1     # 270 windows
# (after the CPU plan step: cache_esmfold_features.py ... --plan-only). Resumable: existing shards are skipped.
#SBATCH -J cache_mdcath
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 05:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/cache_mdcath_%A_%a.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts
SPLIT=$1; WINDOWS=$2; NTASKS=$3
nvidia-smi --query-gpu=name --format=csv,noheader; git log --oneline -1
python investigation/scripts/cache_esmfold_features.py --split $SPLIT --windows $WINDOWS --esmfold-mode eval \
    --seed 0 --out $HOME/scratch/esmdyn_cache/mdcath_$SPLIT --task $SLURM_ARRAY_TASK_ID --n-tasks $NTASKS
