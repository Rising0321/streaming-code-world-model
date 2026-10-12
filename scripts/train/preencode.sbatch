#!/bin/bash
#SBATCH -J cwm-preencode
#SBATCH -p main
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 08:00:00
# Pre-encode every captioned clip, one worker per GPU (visual stage, then text); --ntasks-per-node=16
# --cpus-per-task=8 runs two workers per GPU (the Qwen text encoder is CPU-bound and leaves the GPU ~35% busy).
# Extra arguments go to
# rtcwm.train.preencode, e.g. a helper job meeting the main one from the other end:
#   sbatch scripts/train/preencode.sbatch --stage text --reverse --only-incomplete
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
LOG=$RTCWM_DATA/logs/preencode_$SLURM_JOB_ID
mkdir -p "$LOG"
ARGS=${*:---stage both}
srun --kill-on-bad-exit=0 bash -c "
  source $REPO/scripts/env.sh; use_env cwm; export CUDA_VISIBLE_DEVICES=\$((SLURM_LOCALID * 8 / SLURM_NTASKS_PER_NODE))
  python -m rtcwm.train.preencode --shard $ARGS > $LOG/w\$SLURM_PROCID.log 2>&1"
