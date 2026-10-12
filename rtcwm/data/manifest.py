"""Render manifests: one JSON row per clip (name, map, seed, seconds).

    python -m rtcwm.data.manifest --n 400 --seed0 30000 --maps Town01_Opt Town03_Opt ... > manifest.jsonl

dataset_v1 (460 rows, seeds 20000-20459, 38.9 s) cycled through Town01/02/03/04/05/10HD with Town02 rows
remapped to other maps (no out-and-back routes there); its manifest is data_lists/dataset_v1_manifest.jsonl.
Town04 has since been dropped as well (navmesh under the terrain).
"""
from __future__ import annotations

import argparse
import json

DEFAULT_MAPS = ["Town01_Opt", "Town03_Opt", "Town05_Opt", "Town10HD_Opt"]
CLIP_SECONDS = 934 / 24                # 10 CWM windows of 124 frames at stride 90


def rows(n: int, seed0: int, maps=DEFAULT_MAPS, seconds: float = CLIP_SECONDS, first: int = 0):
    for i in range(first, first + n):
        yield dict(name=f"clip_{i:04d}", map=maps[i % len(maps)], seed=seed0 + i, seconds=seconds)


def read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--first", type=int, default=0, help="index of the first clip (clip_NNNN numbering)")
    ap.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    ap.add_argument("--seconds", type=float, default=CLIP_SECONDS)
    a = ap.parse_args()
    for r in rows(a.n, a.seed0, a.maps, a.seconds, a.first):
        print(json.dumps(r))


if __name__ == "__main__":
    main()
