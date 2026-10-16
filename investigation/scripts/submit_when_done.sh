#!/bin/bash
# Submit follow-on training jobs once a run has finished (its DONE file exists).
#
#   setsid nohup bash investigation/scripts/submit_when_done.sh \
#       ~/scratch/esmdyn_retrain/runs/opt1_fixed_s0 retrain_opt1 retrain_opt2 \
#       "--job-name=retrain_opt2 investigation/slurm/train_from_v2.sbatch all fixed 0" \
#       "--job-name=retrain_opt2 investigation/slurm/train_from_v2.sbatch all bug 0" \
#       > ~/scratch/esmdyn_retrain/logs/after_opt1.log 2>&1 < /dev/null &
#
# Runs on the LOGIN node, from the investigation worktree. Arguments: the run directory to
# wait on, the Slurm job name of its chain (train_from_v2.sbatch resubmits under the same
# name), the job name the follow-on jobs will use (for their W&B live sync), then one quoted
# sbatch argument string per job to submit.
#
# A job id dependency cannot express this: each 8 h attempt exits 0 after resubmitting itself,
# so `afterok` would fire after the first attempt. Instead: poll every 5 minutes; when DONE
# appears, submit every job, then hand over to wandb_live_sync.sh for the new jobs' name. If
# the chain leaves the queue without DONE (a failure: the sbatch does not resubmit after one),
# submit nothing and exit 1.
RUN_DIR=${1:?run directory}; WATCH=${2:?job name to wait on}; NEXT=${3:?job name of the follow-on jobs}
shift 3
[ $# -gt 0 ] || { echo "nothing to submit" >&2; exit 2; }
cd "$(dirname "$0")/../.."                      # the worktree root, where the sbatch paths resolve

log() { echo "[$(date +%FT%T)] $*"; }
log "waiting for $RUN_DIR/DONE (chain $WATCH); then: $*"
gone=0
while [ ! -f "$RUN_DIR/DONE" ]; do
    if squeue -u $USER -h -n $WATCH 2>/dev/null | grep -q .; then
        gone=0
    else
        gone=$((gone + 1))                       # twice in a row: not a resubmission gap
        if [ $gone -ge 2 ] && [ ! -f "$RUN_DIR/DONE" ]; then
            log "$WATCH left the queue without $RUN_DIR/DONE: not submitting"
            exit 1
        fi
    fi
    sleep 300
done
log "DONE: $(cat $RUN_DIR/DONE)"
for args in "$@"; do
    # word-splitting of each quoted job string is intended
    jid=$(sbatch --parsable $args) && log "submitted $jid: sbatch $args" || log "FAILED: sbatch $args"
done
log "starting W&B live sync for $NEXT"
exec bash investigation/scripts/wandb_live_sync.sh $NEXT
