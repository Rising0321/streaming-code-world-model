#!/bin/bash
#SBATCH -J cwm-release
#SBATCH -p main
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 04:00:00
# The released CWM sampler (bidirectional windows, Retake34) on configs from `rtcwm.eval.open_world release`,
# one config directory per GPU.  usage: sbatch -o LOG scripts/eval/release_cwm.sbatch DIR/example_04 DIR/example_08 ...
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
use_env cwm
i=0
for d in "$@"; do
  ( CUDA_VISIBLE_DEVICES=$i python -m cwm_h3_inference prepare --config "$d/config.json" && \
    CUDA_VISIBLE_DEVICES=$i python -m cwm_h3_inference generate --config "$d/config.json" ) > "$d/run.log" 2>&1 &
  i=$((i + 1))
done
wait
