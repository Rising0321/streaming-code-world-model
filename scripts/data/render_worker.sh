#!/bin/bash
# One render worker per GPU (launched by render_dataset.sbatch): worker W renders manifest rows
# W, W+NT, ... into $RTCWM_DATA/<dataset>/<name>, then compiles the coarse proxy. A clip that is
# complete (QC pass + coarse proxy) is skipped, so the job can simply be resubmitted.
# usage: render_worker.sh MANIFEST.jsonl DATASET_DIR
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
MAN=$1; DST=$2
W=$SLURM_PROCID; NT=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID     # whole node allocated: pin this worker's server and CUDA work
use_env carla
complete() { python -c "import sys; from rtcwm.data.clip import complete; sys.exit(0 if complete('$1') else 1)"; }
N=$(wc -l < "$MAN")
echo "worker $W/$NT on $(hostname) gpu $CUDA_VISIBLE_DEVICES"
for (( r=W; r<N; r+=NT )); do
  read NAME MAP SEED SEC < <(sed -n "$((r+1))p" "$MAN" | python -c "import json,sys; d=json.load(sys.stdin); print(d['name'], d['map'], d['seed'], d['seconds'])")
  OUT=$DST/$NAME
  if complete "$OUT"; then echo "DONE already $OUT"; continue; fi
  bash "$REPO/scripts/data/render_clip.sh" "$OUT" --map "$MAP" --seconds "$SEC" --seed "$SEED" || { echo "RENDER_FAILED $OUT"; continue; }
  use_env carla
  python -m rtcwm.data.proxy --workers 16 "$OUT" || { echo "COARSE_FAILED $OUT"; continue; }
  complete "$OUT" && echo "DONE $OUT" || echo "INCOMPLETE $OUT"
done
echo "WORKER_FINISHED $W"
