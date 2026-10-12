"""python -m rtcwm.data.qc {clipping,audit,coarse} [--workers N] CLIP_DIR ... | lists"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from ... import paths
from . import audit, clipping, coarse, lists

SCANS = dict(clipping=clipping.scan, audit=audit.audit, coarse=coarse.scan)


def _run(job):
    check, clip = job
    try:
        rep = SCANS[check](Path(clip))
    except Exception as e:                                      # report and keep going
        return f"{Path(clip).name} {check} ERROR {e!r}"
    out = paths.QC / check
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(clip).name}.json").write_text(json.dumps(rep))
    return f"{Path(clip).name} {check} bad_windows={rep.get('bad_windows', '-')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", choices=[*SCANS, "lists"])
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_intermixed_args()          # options may sit between the check name and the clips
    if a.check == "lists":
        print(lists.build())
        return
    jobs = [(a.check, c) for c in a.clips]
    if a.workers > 1:
        with Pool(a.workers) as p:
            for line in p.imap_unordered(_run, jobs):
                print(line, flush=True)
    else:
        for j in jobs:
            print(_run(j), flush=True)


if __name__ == "__main__":
    main()
