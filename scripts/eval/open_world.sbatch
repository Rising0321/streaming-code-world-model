#!/bin/bash
#SBATCH -J cwm-openworld
#SBATCH -p main
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 04:00:00
# Released open-world examples (rtcwm.eval.open_world): prepare on GPU 0, then one example per GPU, for each LoRA.
# usage: sbatch -o LOG scripts/eval/open_world.sbatch OUT_DIR "example_04 example_08 ..." LORA [LORA ...]
#        (OUT_DIR/<lora stem>/<example>_stream.mp4)
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
use_env cwm
cd "$REPO"
OUT=$1; EXAMPLES=$2; shift 2
CUDA_VISIBLE_DEVICES=0 python -m rtcwm.eval.open_world prepare --examples $EXAMPLES ${OW_ARGS:-} || exit 1
i=0
for e in $EXAMPLES; do
  ( for lora in "$@"; do
      d="$OUT/$(basename "$(dirname "$lora")")_$(basename "$lora" .safetensors)"; mkdir -p "$d"
      CUDA_VISIBLE_DEVICES=$i python -m rtcwm.eval.open_world generate --lora "$lora" --out "$d" --examples $e ${OW_ARGS:-} > "$d/$e.log" 2>&1
    done ) &
  i=$((i + 1))
done
wait
