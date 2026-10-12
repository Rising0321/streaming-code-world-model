"""Render-time quality control.

A clip passes when, for no more than 0.5 s at a time, the protagonist is hidden, the camera is
pressed into geometry, the protagonist's legs are inside solid scenery, it is buried in the ground,
stands in a hedge / planter, or something stands between it and the camera; when no pole passes
through its body (>= 3 frames); when the route was completed (got well away, came back); and when the
image is not too dark. The interpenetration checks are rtcwm.data.qc.clipping, run on the written clip.
"""
from __future__ import annotations

import numpy as np

from ..geometry import longest_run
from ..qc import clipping
from .config import FPS

MAX_BAD_S = 0.5
MIN_LUMA = 60
MAX_PIERCED_FRAMES = 3


class RenderQC:
    def __init__(self):
        self.hero_frac, self.near_frac, self.leg_clip, self.luma = [], [], [], []

    def add(self, hero_frac: float, near_frac: float, leg_clip: float, luma: float | None):
        self.hero_frac.append(hero_frac)
        self.near_frac.append(near_frac)
        self.leg_clip.append(leg_clip)
        if luma is not None:
            self.luma.append(luma)

    def summary(self, traj, leg: float, speed: float, seconds: float) -> dict:
        t = np.asarray(traj)
        d0 = np.linalg.norm(t[:, :2] - t[0, :2], axis=1)
        third = len(d0) // 3
        qc = dict(max_hidden_s=longest_run(np.array(self.hero_frac) < 0.003) / FPS,
                  max_near_frac=float(np.max(self.near_frac)),
                  max_near_s=longest_run(np.array(self.near_frac) > 0.30) / FPS,
                  path_m=float(np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1).sum()),
                  farthest_first_third_m=float(d0[:third + 1].max()),
                  closest_return_m=float(d0[third:2 * third + 1].min()),
                  hero_z_range_m=float(t[:, 2].max() - t[:, 2].min()),
                  mean_luma=float(np.mean(self.luma)),
                  max_clip_s=longest_run(np.array(self.leg_clip) > 0.25) / FPS)
        qc["route_ok"] = bool(qc["path_m"] >= 0.6 * speed * seconds and qc["farthest_first_third_m"] >= 0.5 * leg
                              and qc["closest_return_m"] <= 6.0)
        qc["pass"] = bool(qc["route_ok"] and qc["max_hidden_s"] <= MAX_BAD_S and qc["max_near_s"] <= MAX_BAD_S
                          and qc["hero_z_range_m"] <= 0.3 and qc["mean_luma"] >= MIN_LUMA and qc["max_clip_s"] <= MAX_BAD_S)
        return qc


def add_clipping(qc: dict, clip_dir) -> dict:
    """Run the interpenetration scan on the written clip and fold it into the verdict."""
    rep = clipping.scan(clip_dir)
    qc["max_buried_s"] = rep["max_buried_run"] / FPS
    qc["max_occluded_s"] = rep["max_occluded_run"] / FPS
    qc["max_inside_s"] = rep["max_inside_run"] / FPS
    qc["max_pierced_frames"] = rep["max_pierced_run"]
    qc["pass"] = bool(qc["pass"] and qc["max_buried_s"] <= MAX_BAD_S and qc["max_occluded_s"] <= MAX_BAD_S
                      and qc["max_inside_s"] <= MAX_BAD_S and qc["max_pierced_frames"] < MAX_PIERCED_FRAMES)
    return qc
