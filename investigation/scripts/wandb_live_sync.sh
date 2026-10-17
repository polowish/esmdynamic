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
# live, and then marked done; nothing is re-uploaded. It exits once no job whose name STARTS
# with the given name (`retrain` covers retrain_opt1 and retrain_opt2; resubmissions keep their
# name) is queued and every folder is done.
#
# The login node runs logind with KillUserProcesses=yes: everything started in an ssh session,
# setsid/nohup included, is killed when that session ends. Compute nodes have no route to the
# proxy (checked 2026-10-16), so this cannot move into Slurm. Restart it after reconnecting;
# it is restart-safe (state files; W&B drops already-uploaded history steps).
#
# State is one small file per folder in live_sync/ (<folder>.pid, .rc, .done, .out), so a
# restarted loop picks up where the last one was. The API key is read from its file into the
# environment, never onto a command line or into the log. Every wandb directory is on scratch.
JOB_NAME=${1:?job name, e.g. retrain_opt1}
ENTITY=sophia-madejski-university-of-california
ROOT=$HOME/scratch/esmdyn_retrain/wandb/wandb
STATE=$HOME/scratch/esmdyn_retrain/wandb/live_sync

module load CBI miniforge3 >/dev/null 2>&1
conda activate $HOME/scratch/venvs/esmdynamic
set -o pipefail     # no -u: conda's activate scripts read unset variables
export PYTHONPATH=$HOME/scratch/pylibs/wandb
export WANDB_CONFIG_DIR=$HOME/scratch/tool_caches/wandb/config
export WANDB_CACHE_DIR=$HOME/scratch/tool_caches/wandb/cache
export WANDB_DATA_DIR=$HOME/scratch/tool_caches/wandb/data
WANDB_API_KEY=$(cat $HOME/scratch/tool_caches/wandb/api_key); export WANDB_API_KEY
mkdir -p $STATE

log() { echo "[$(date +%FT%T)] $*"; }
log "watching $ROOT for $JOB_NAME"
while true; do
    active=0
    for d in $ROOT/offline-run-*; do
        [ -d "$d" ] || continue
        s=$STATE/$(basename $d)
        [ -f $s.done ] && continue
        if [ -f $s.rc ]; then                          # its sync has ended
            rc=$(cat $s.rc)
            log "live sync end: $(basename $d) (exit $rc)"
            if [ "$rc" = 0 ]; then touch $s.done; else rm -f $s.rc $s.pid; fi   # failed: retry
            continue
        fi
        if [ -f $s.pid ] && kill -0 $(cat $s.pid) 2>/dev/null; then
            active=$((active + 1)); continue           # still streaming
        fi
        log "live sync start: $(basename $d)"
        ( timeout 9h python -m wandb beta sync --live -e $ENTITY "$d" >> $s.out 2>&1; echo $? > $s.rc ) &
        echo $! > $s.pid
        active=$((active + 1))
    done
    if [ $active -eq 0 ] && ! squeue -u $USER -h -o %j 2>/dev/null | grep -q "^$JOB_NAME"; then
        log "no $JOB_NAME in the queue and nothing left to sync: exiting"
        break
    fi
    sleep 60
done
