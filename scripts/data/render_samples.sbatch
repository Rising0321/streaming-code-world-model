#!/bin/bash
#SBATCH -J a-render-samples
#SBATCH -p main
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 03:00:00
# Render up to 8 sample clips (one per GPU) to review renderer changes, then their coarse proxy and
# RGB | coarse proxy | overlay previews.
# usage: sbatch -o LOG scripts/data/render_samples.sbatch OUT_DIR MANIFEST.jsonl
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
OUT=$1; MAN=$2
mkdir -p "$OUT"
i=0
while read -r line; do
  read NAME MAP SEED SEC < <(echo "$line" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['name'], d['map'], d['seed'], d['seconds'])")
  CUDA_VISIBLE_DEVICES=$i bash "$REPO/scripts/data/render_clip.sh" "$OUT/$NAME" --map "$MAP" --seconds "$SEC" --seed "$SEED" \
    > "$OUT/$NAME.render.log" 2>&1 &
  i=$((i + 1))
done < "$MAN"
wait
grep -h "^OK \|failed" "$OUT"/*.render.log
use_env carla
DONE=$(for d in "$OUT"/*/; do [ -f "$d/meta.json" ] && echo -n "$d "; done)
python -m rtcwm.data.proxy --workers 64 $DONE
python -m rtcwm.data.preview "$OUT/preview" $DONE --proxy conditions_coarse
