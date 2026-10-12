#!/bin/bash
#SBATCH -J cwm-distill
#SBATCH -p main
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 24:00:00
# Few-step DMD distillation of a causal DF checkpoint, 8 nodes x 8 GPUs by default.
# usage: sbatch [--exclude=$RTCWM_EXCLUDE] scripts/train/distill.sbatch OUT_DIR STUDENT_LORA [rtcwm.train.distill args]
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
OUT=$1; STUDENT=$2; shift 2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29512
# no expandable_segments: near-full memory, its retries unmap/remap whole segments (a rollout chunk 6 s -> 60 s)
srun --kill-on-bad-exit=1 bash -c "source $REPO/scripts/env.sh; use_env cwm; cd $REPO; \
  torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 --node-rank=\$SLURM_NODEID \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID \
  -m rtcwm.train.distill --out $OUT --student-lora $STUDENT $*"
