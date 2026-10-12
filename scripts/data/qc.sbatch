#!/bin/bash
#SBATCH -J a-qc
#SBATCH -p cpuonly
#SBATCH --qos=cpuonly
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=200G
#SBATCH -t 03:00:00
# All QC scans over a dataset, then the data lists training reads.
# usage: sbatch -o LOG scripts/data/qc.sbatch DATASET_DIR
source "$SLURM_SUBMIT_DIR/scripts/env.sh"
use_env cwm
CLIPS=$(python -c "
from pathlib import Path; from rtcwm.data.clip import complete
print(' '.join(str(d) for d in sorted(Path('$1').glob('clip_*')) if complete(d)))")
for check in clipping coarse audit; do
  python -m rtcwm.data.qc $check --workers 60 $CLIPS
done
python -m rtcwm.data.qc lists
