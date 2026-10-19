#!/bin/bash
# Plan 05 step 4: fine-tune from the released V2 heads on the cached mdCATH windows.
#   sbatch investigation/slurm/train_from_v2.sbatch at-init fixed 0     # option 1
#   sbatch investigation/slurm/train_from_v2.sbatch all fixed 0         # option 2
#   sbatch investigation/slurm/train_from_v2.sbatch all bug 0           # option 2's control
#   sbatch investigation/slurm/train_from_v2.sbatch all fixed 0 <tag> [trainer options...]
#       a variant: the tag goes into the run name (v2ft-all-heads-fixed-<tag>-s0) and the options
#       are passed to train_from_v2.py unchanged, on every resubmission too
# (from the investigation worktree). Out: ~/scratch/esmdyn_retrain/runs/v2ft-{dcm-only,all-heads}-<arm>-s<seed>.
#
# L40S only (user, 2026-10-15): the cache was written on L40S, and anything that runs ESMFold
# live later (test-set evaluation) runs there too. 8 h jobs backfill far better than one long
# one at low fair-share priority. The trainer stops after the last epoch that fits
# (--time-limit-h), and this script resubmits itself with --resume until the run writes DONE
# (early stop or the epoch cap). It does NOT resubmit after a failure (non-zero exit), or if
# an attempt made no progress, so a crash cannot loop.
#SBATCH -J retrain
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/remote/smadejski/scratch/esmdyn_retrain/logs/retrain_%x_%j.out
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"
set -uo pipefail   # after activation (conda's cuda-nvcc activate script reads unset variables)
cd $HOME/scratch/github/esmdynamic-investigation
export PYTHONPATH=$PWD:$PWD/investigation/scripts:$HOME/scratch/pylibs/wandb
export WANDB_DIR=$HOME/scratch/esmdyn_retrain/wandb      # offline; synced from the login node
mkdir -p $WANDB_DIR

TRAINABLE=$1; ARM=$2; SEED=${3:-0}; TAG=${4:-}
shift $(( $# < 4 ? $# : 4 ))              # what remains: extra trainer options
EXTRA=("$@")
# run name = output folder = W&B display name: v2ft-<what trains>-<arm>-s<seed>
# (v2ft: fine-tuned from the released V2 heads; option 1 = dcm-only, option 2 = all-heads)
case $TRAINABLE in at-init) SCOPE=dcm-only;; all) SCOPE=all-heads;; *) echo "trainable: at-init|all" >&2; exit 2;; esac
OUT=$HOME/scratch/esmdyn_retrain/runs/v2ft-${SCOPE}-${ARM}${TAG:+-$TAG}-s${SEED}
C=$HOME/scratch/esmdyn_cache
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git log --oneline -1
epoch_now() { python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['state']['epoch'])" "$OUT/last.pt" 2>/dev/null || echo -1; }
before=$(epoch_now)

python -u investigation/scripts/train_from_v2.py --trainable $TRAINABLE --arm $ARM --seed $SEED \
    --train-cache $C/mdcath_train --val-cache $C/mdcath_val --out $OUT --resume \
    --time-limit-h 7.6 --wandb offline ${EXTRA[@]+"${EXTRA[@]}"}
rc=$?
after=$(epoch_now)
echo "exit $rc; epoch $before -> $after"
if [ $rc -ne 0 ]; then echo "failed: not resubmitting" >&2; exit $rc; fi
if [ -f $OUT/DONE ]; then echo "DONE: $(cat $OUT/DONE)"; exit 0; fi
if [ "$after" -le "$before" ]; then echo "no progress this attempt: not resubmitting" >&2; exit 1; fi
sbatch --job-name=$SLURM_JOB_NAME investigation/slurm/train_from_v2.sbatch $TRAINABLE $ARM $SEED "$TAG" ${EXTRA[@]+"${EXTRA[@]}"}
