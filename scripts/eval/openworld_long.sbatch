#!/bin/bash
#SBATCH -J cwm-openworld-long
#SBATCH -p main
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 06:00:00
# Long open-world streams: compile open-code-world-model gallery scenes (proxy from the same renderer and rig as
# our CARLA proxies) for WINDOWS released windows, then stream them with one LoRA, one scene per GPU.
# usage: WINDOWS=16 OW_ARGS="--chunks 85 --steps 4 --rule x0 --shift 5 --kv-reuse" \
#        sbatch -o LOG scripts/eval/openworld_long.sbatch SCENE_DIR OUT_DIR LORA
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
use_env cwm
SCENES=$1; OUT=$2; LORA=$3
OCWM=$HOME/projects/open-code-world-model
mkdir -p "$SCENES" "$OUT"
# a scene whose scripted actions break the compiler's invariants at this length is not written; stream the rest
( cd "$OCWM" && CWM_EXAMPLES=$REPO/runs/openworld/examples40 CWM_GALLERY_WINDOWS=${WINDOWS:-16} CWM_GALLERY_OUT=$SCENES \
    CWM_WORKERS=16 python examples/gallery.py )
cd "$REPO"
NAMES=$(for d in "$SCENES"/*/; do [ -d "$d/conditions" ] && [ -f "$d/first_frame.png" ] && basename "$d"; done)
[ -n "$NAMES" ] || { echo "no scene compiled"; exit 1; }
echo "scenes: $NAMES"
CUDA_VISIBLE_DEVICES=0 python -m rtcwm.eval.open_world prepare --root "$SCENES" --examples $NAMES ${OW_ARGS:-} || exit 1
i=0
for e in $NAMES; do
  CUDA_VISIBLE_DEVICES=$i python -m rtcwm.eval.open_world generate --root "$SCENES" --lora "$LORA" --out "$OUT" --examples $e ${OW_ARGS:-} > "$OUT/$e.log" 2>&1 &
  i=$((i + 1))
done
wait
