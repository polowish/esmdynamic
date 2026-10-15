#!/bin/bash
# Plan 05: verify the ESMFold window caches before training (investigation/scripts/verify_cache.py).
#   sbatch -p cpu -c 16 --mem=64G investigation/slurm/verify_cache.sbatch integrity
#   sbatch -p gpu --gres=gpu:nvidia_l40s:1        investigation/slurm/verify_cache.sbatch live
#   sbatch -p gpu --gres=gpu:nvidia_rtx_pro_6000:1 investigation/slurm/verify_cache.sbatch live
#   sbatch -p gpu --gres=gpu:<type>:1 investigation/slurm/verify_cache.sbatch precision   # bf16 vs fp32
# integrity resumes from <report>.progress.jsonl if cut short; give it -t 06:00:00.
# Checks mdcath_train and mdcath_val; reports land in ~/scratch/esmdyn_retrain/verify/.
#SBATCH -J verify_cache
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/verify_cache_%j.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts
MODE=$1; shift
OUT=$HOME/scratch/esmdyn_retrain/verify
git log --oneline -1
[ "$MODE" != integrity ] && nvidia-smi --query-gpu=name --format=csv,noheader
rc=0
for split in train val; do
    tag=$MODE; [ "$MODE" = live ] && tag=live_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _ | tr -cd '[:alnum:]_')
    n=48; [ "$split" = val ] && n=16
    args=(); [ "$MODE" = live ] && args=(-n $n)
    if [ "$MODE" = precision ]; then
        [ "$split" = val ] && continue                     # train has both writers
        tag=precision_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _ | tr -cd '[:alnum:]_')
    fi
    python -u investigation/scripts/verify_cache.py --cache $HOME/scratch/esmdyn_cache/mdcath_$split \
        --report $OUT/${split}_${tag}.json $MODE "${args[@]}" "$@" || rc=1
done
exit $rc
