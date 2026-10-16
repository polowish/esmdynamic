#!/bin/bash
# Keep a training run's W&B page current while it trains (compute nodes are offline).
#
#   setsid nohup bash investigation/scripts/wandb_live_sync.sh retrain_opt1 \
#       > ~/scratch/esmdyn_retrain/logs/wandb_live_sync_retrain_opt1.log 2>&1 < /dev/null &
#
# Runs on the LOGIN node. Every minute it looks for offline run folders under
# ~/scratch/esmdyn_retrain/wandb/wandb/ and streams each new one with `wandb beta sync --live`
# (which tails the .wandb file until the run finishes). Each 8 h attempt of
# train_from_v2.sbatch resumes the same run id in a new folder, so a folder is synced once,
# live, and then marked done; nothing is re-uploaded. It exits after the Slurm job name
# (resubmissions keep it) has left the queue and every folder is done.
#
# The API key is read from its file into the environment, never onto a command line or into
# the log. Every wandb directory is on scratch.
set -uo pipefail
JOB_NAME=${1:?job name, e.g. retrain_opt1}
ENTITY=sophia-madejski-university-of-california
ROOT=$HOME/scratch/esmdyn_retrain/wandb/wandb
DONE=$HOME/scratch/esmdyn_retrain/wandb/live_synced.txt

module load CBI miniforge3 >/dev/null 2>&1
conda activate $HOME/scratch/venvs/esmdynamic
export PYTHONPATH=$HOME/scratch/pylibs/wandb
export WANDB_CONFIG_DIR=$HOME/scratch/tool_caches/wandb/config
export WANDB_CACHE_DIR=$HOME/scratch/tool_caches/wandb/cache
export WANDB_DATA_DIR=$HOME/scratch/tool_caches/wandb/data
WANDB_API_KEY=$(cat $HOME/scratch/tool_caches/wandb/api_key); export WANDB_API_KEY
touch $DONE
declare -A live            # folder -> pid of its live sync

log() { echo "[$(date +%FT%T)] $*"; }
log "watching $ROOT for $JOB_NAME"
while true; do
    for d in $ROOT/offline-run-*; do
        [ -d "$d" ] || continue
        b=$(basename $d)
        grep -qx "$b" $DONE && continue
        if [ -z "${live[$b]:-}" ]; then
            log "live sync start: $b"
            timeout 9h python -m wandb beta sync --live -e $ENTITY "$d" >> $DONE.$b.out 2>&1 &
            live[$b]=$!
        elif ! kill -0 ${live[$b]} 2>/dev/null; then
            wait ${live[$b]}; rc=$?
            log "live sync end: $b (exit $rc)"
            unset "live[$b]"
            [ $rc -eq 0 ] && echo "$b" >> $DONE      # failed: retried next pass
        fi
    done
    if [ ${#live[@]} -eq 0 ] && ! squeue -u $USER -h -n $JOB_NAME 2>/dev/null | grep -q .; then
        log "no $JOB_NAME in the queue and nothing left to sync: exiting"
        break
    fi
    sleep 60
done
