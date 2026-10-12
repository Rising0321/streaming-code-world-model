#!/bin/bash
#SBATCH -J cwm-train
#SBATCH -p main
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 14:00:00
# Causal LoRA training, 8 nodes x 8 GPUs by default.
# usage: sbatch [--exclude=$RTCWM_EXCLUDE] scripts/train/train.sbatch OUT_DIR [rtcwm.train.train args]
#   e.g. sbatch scripts/train/train.sbatch $RTCWM_DATA/runs/df_v2 --objective df --epochs 2
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
OUT=$1; shift
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29511
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --kill-on-bad-exit=1 bash -c "source $REPO/scripts/env.sh; use_env cwm; cd $REPO; \
  torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 --node-rank=\$SLURM_NODEID \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID \
  -m rtcwm.train.train --out $OUT $*"
