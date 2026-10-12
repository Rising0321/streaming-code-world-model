#!/bin/bash
#SBATCH -J cwm-caption
#SBATCH -p main
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 08:00:00
# Captions for every complete clip of a dataset, one Qwen3-VL worker per GPU (clips W, W+NT, ...).
# usage: sbatch scripts/data/caption.sbatch DATASET_DIR        (CAPTION_ARGS=--static: one static caption per clip)
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
LOG=$RTCWM_DATA/logs/caption_$SLURM_JOB_ID
mkdir -p "$LOG"
srun --kill-on-bad-exit=0 bash -c "
  source $REPO/scripts/env.sh; use_env cwm; export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
  CLIPS=\$(python -c \"
import sys; from pathlib import Path; from rtcwm.data.clip import complete
cl = [str(d) for d in sorted(Path('$1').glob('clip_*')) if complete(d)]
print(' '.join(cl[\$SLURM_PROCID::\$SLURM_NTASKS]))\")
  python -m rtcwm.data.captions ${CAPTION_ARGS:-} \$CLIPS > $LOG/w\$SLURM_PROCID.log 2>&1"
