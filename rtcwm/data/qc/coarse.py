"""Coarse-proxy failures: the frame swallowed by vegetation (coarse > 60% vegetation where the fine
proxy has < 30%), or the protagonist visible in the fine proxy (> 1% of the frame) but nearly absent
from the coarse one (< 20% of that). A window is bad with >= 3 such samples (every 6th frame)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..clip import n_windows, read_meta, read_proxy, window_frames
from ..geometry import CLS_HERO, CLS_VEGETATION

SAMPLE_EVERY = 6
MIN_BAD_SAMPLES = 3


def frame_bad(clip: Path, i: int, kind: str = "conditions_coarse") -> bool:
    _, sf = read_proxy(clip, i)
    _, sc = read_proxy(clip, i, kind)
    engulfed = (sc == CLS_VEGETATION).mean() > 0.6 and (sf == CLS_VEGETATION).mean() < 0.3
    hf, hc = (sf == CLS_HERO).mean(), (sc == CLS_HERO).mean()
    return bool(engulfed or (hf > 0.01 and hc < 0.2 * hf))


def scan(clip: Path, kind: str = "conditions_coarse") -> dict:
    clip = Path(clip)
    n = read_meta(clip)["frames"]
    idx = list(range(0, n, SAMPLE_EVERY))
    bad = {i: frame_bad(clip, i, kind) for i in idx}
    windows = []
    for w in range(n_windows(n)):
        sl = window_frames(w)
        k = sum(bad[i] for i in idx if sl.start <= i < sl.stop)
        windows.append(dict(w=w, bad_samples=int(k), bad=k >= MIN_BAD_SAMPLES))
    return dict(clip=clip.name, frames=n, bad_frames=[i for i in idx if bad[i]],
                bad_windows=[x["w"] for x in windows if x["bad"]], windows=windows)
