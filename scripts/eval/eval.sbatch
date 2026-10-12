#!/bin/bash
#SBATCH -J cwm-eval
#SBATCH -p main
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH -t 06:00:00
# Stream up to 4 held-out clips in parallel (one per GPU) and compare with the ground truth.
# usage: sbatch -o LOG scripts/eval/eval.sbatch LORA OUT_DIR clip [clip ...]
#        EVAL_ARGS="--chunks 12" for a short stream; "--steps 4 --rule x0 --shift 5" for a distilled student
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
use_env cwm
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LORA=$1; OUT=$2; shift 2
mkdir -p "$OUT"
i=0
for c in "$@"; do
  CUDA_VISIBLE_DEVICES=$i python -m rtcwm.eval.evaluate --lora "$LORA" --out "$OUT" --clips "$c" ${EVAL_ARGS:-} \
    > "$OUT/$c.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h '"pixel_mse"' "$OUT"/*.log
