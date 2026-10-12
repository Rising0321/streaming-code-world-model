#!/bin/bash
# usage: render_clip.sh OUT [rtcwm.data.render args...]
# Starts a CARLA server on a free random port, renders one clip, stops it. Up to 4 attempts: a server
# crash (VK_ERROR_DEVICE_LOST is seen intermittently) or a QC rejection (exit 3, retried with the seed
# offset by 1000) starts over. A clip only counts as done when meta.json exists.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
OUT=$1; shift
use_env carla
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
CARLA=${CARLA_ROOT:-$RTCWM_DATA/CARLA_0.9.15}
[ -f "$OUT/meta.json" ] && { echo "already done $OUT"; exit 0; }
for attempt in 1 2 3 4; do
  # the server also uses PORT+1; keep PORT well under 65535 (an out-of-range port segfaults libcarla)
  PORT=$(( 20000 + (RANDOM % 3000) * 10 ))
  rm -rf "$OUT"; mkdir -p "$OUT"
  # Vulkan sees every GPU on the node and ignores CUDA_VISIBLE_DEVICES: pin UE to the assigned GPU
  $CARLA/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=$PORT -quality-level=Epic \
    -graphicsadapter=${CUDA_VISIBLE_DEVICES%%,*} > "$OUT/server.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 90); do
    python -c "import carla; c=carla.Client('localhost',$PORT); c.set_timeout(5); c.get_server_version()" 2>/dev/null && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 4
  done
  RC=1
  if [ $ok = 1 ]; then
    CARLA_SEED_OFFSET=$(( (attempt - 1) * 1000 )) timeout 3600 python -m rtcwm.data.render --port $PORT --out "$OUT" "$@"; RC=$?
  fi
  pkill -f "carla-rpc-port=$PORT" 2>/dev/null; kill $SRV 2>/dev/null; sleep 5
  [ $RC = 0 ] && [ -f "$OUT/meta.json" ] && { echo "OK $OUT (attempt $attempt)"; exit 0; }
  [ $RC = 3 ] && echo "attempt $attempt rejected by QC (rc=3), retrying with a new seed"
  echo "attempt $attempt failed (rc=$RC):"; grep -m2 -E "Fatal|ERROR_DEVICE|Segmentation" "$OUT/server.log"
done
exit 1
