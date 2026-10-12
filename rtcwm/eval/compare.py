"""Side-by-side comparison of evaluation runs: [run A gen | run B gen | ... | GT] per sample, plus a
summary table of pixel MSE.

    python -m rtcwm.eval.compare --runs EVAL_DIR_A EVAL_DIR_B --labels "step 0" "step 49" --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .video import read_mp4, write_mp4

W = 672


def _label(im, text):
    im = im.copy()
    cv2.putText(im, text, (10, 30), 0, 0.9, (0, 0, 0), 5)
    cv2.putText(im, text, (10, 30), 0, 0.9, (255, 255, 255), 2)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert len(a.runs) == len(a.labels)
    a.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name.replace("_gen_vs_gt.mp4", "") for p in a.runs[0].glob("*_gen_vs_gt.mp4"))
    table = []
    for n in names:
        vids = [read_mp4(r / f"{n}_gen_vs_gt.mp4") for r in a.runs]
        half = vids[0].shape[2] // 2
        cols = [v[..., :half, :] for v in vids] + [vids[0][..., half:, :]]
        labels = list(a.labels) + ["ground truth"]
        h = int(cols[0].shape[1] * W / half)
        frames = [np.concatenate([_label(cv2.resize(c[i], (W, h)), lab) for c, lab in zip(cols, labels)], 1)
                  for i in range(min(len(c) for c in cols))]
        write_mp4(a.out / f"{n}_compare.mp4", np.stack(frames), crf=22)
        row = dict(sample=n)
        for r, lab in zip(a.runs, a.labels):
            f = r / f"{n}.json"
            row[lab] = json.loads(f.read_text())["pixel_mse"] if f.exists() else None
        table.append(row)
        print(json.dumps(row), flush=True)
    (a.out / "summary.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
