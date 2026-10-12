#!/bin/bash
#SBATCH -J a-render
#SBATCH -p main
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 12:00:00
# Render a manifest: one job, whole nodes, one worker per GPU.
# usage: sbatch -o LOGDIR/%j.log scripts/data/render_dataset.sbatch MANIFEST.jsonl DATASET_DIR
# (add --exclude=$RTCWM_EXCLUDE after sourcing scripts/env.sh)
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
LOG=$RTCWM_DATA/logs/render_$SLURM_JOB_ID
mkdir -p "$LOG" "$2"
srun --kill-on-bad-exit=0 bash -c "bash $REPO/scripts/data/render_worker.sh $1 $2 > $LOG/w\$SLURM_PROCID.log 2>&1"
