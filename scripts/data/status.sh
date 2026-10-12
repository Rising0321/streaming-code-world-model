#!/bin/bash
# One-line dataset status: complete clips (QC pass + coarse proxy), clips with captions, running jobs.
# usage: bash scripts/data/status.sh [DATASET_DIR]
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
python3 -c "
from pathlib import Path; from rtcwm import paths; from rtcwm.data.clip import complete, has_prompts
d = Path('${1:-}') if '${1:-}' else paths.DATASET
cl = [c for c in d.glob('clip_*') if c.is_dir()]
print(f'{d.name}: {sum(complete(c) for c in cl)}/{len(cl)} complete, {sum(has_prompts(c) for c in cl)} captioned', end='  ')"
squeue -u "$USER" -h -o "%i:%j:%T:%M" | tr '\n' ' '; echo
