"""The data lists training reads (data_lists/, in git).

    clips.json        {"keep": [...], "dropped": {reason: [...]}}   clips that passed render QC
    bad_windows.json  {clip: [window, ...]}                          windows flagged by any QC report

Maps dropped as a whole: Town02 (no out-and-back routes), Town04 (navmesh under the terrain:
protagonist and camera underground, or walking on highways).
"""
from __future__ import annotations

import json

from ... import paths
from ..clip import complete, read_meta

DROPPED_MAPS = {"Town02_Opt": "no out-and-back sidewalk routes", "Town04_Opt": "navmesh under the terrain"}
CHECKS = ("clipping", "coarse")


def build() -> dict:
    keep, dropped = [], {}
    for d in sorted(paths.DATASET.glob("clip_*")):
        if not d.is_dir() or not complete(d):
            continue
        m = read_meta(d)["map"]
        if m in DROPPED_MAPS:
            dropped.setdefault(DROPPED_MAPS[m], []).append(d.name)
        else:
            keep.append(d.name)
    bad = {}
    for check in CHECKS:
        for c in keep:
            f = paths.QC / check / f"{c}.json"
            if f.exists():
                ws = json.loads(f.read_text())["bad_windows"]
                if ws:
                    bad[c] = sorted(set(bad.get(c, [])) | set(ws))
    paths.DATA_LISTS.mkdir(exist_ok=True)
    (paths.DATA_LISTS / "clips.json").write_text(json.dumps(dict(keep=keep, dropped=dropped), indent=1))
    (paths.DATA_LISTS / "bad_windows.json").write_text(json.dumps(bad, indent=1, sort_keys=True))
    return dict(keep=len(keep), dropped={k: len(v) for k, v in dropped.items()},
                bad_windows=sum(len(v) for v in bad.values()))


def load():
    """(kept clip names, {clip: set(bad windows)})."""
    clips = json.loads((paths.DATA_LISTS / "clips.json").read_text())["keep"]
    bad = json.loads((paths.DATA_LISTS / "bad_windows.json").read_text())
    return clips, {c: set(v) for c, v in bad.items()}
