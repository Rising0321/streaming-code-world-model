#!/bin/bash
#SBATCH -J cwm-smoke
#SBATCH -p main
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 02:00:00
# Tests (positions, masks, sampler vs the training forward, pre-encode vs CWM) on GPU 0, then 6 training
# steps on 8 GPUs. Run before every training submission that changed model code.
# usage: sbatch -o LOG scripts/train/smoke.sbatch [OUT_DIR]          (SKIP_TESTS=1 runs the training steps only)
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
use_env cwm
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${1:-$RTCWM_DATA/runs/smoke_$SLURM_JOB_ID}
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  for t in tests/test_positions.py tests/test_masks.py tests/test_modulation.py tests/test_dense_attention.py tests/test_samplers.py tests/test_kv_reuse.py tests/test_preencode.py; do
    CUDA_VISIBLE_DEVICES=0 python $t || exit 1
  done
fi
torchrun --standalone --nproc-per-node=8 -m rtcwm.train.train --out $OUT/df \
  --steps 6 --warmup 2 --save-every 1000 --val-clips 2 --max-samples 64 || exit 1
echo SMOKE_OK
