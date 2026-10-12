"""Reading a rendered clip directory.

    rgb.mp4                 1344x768, 24 fps
    conditions/             fine proxy (CARLA depth + segmentation), NNNNNN.{depth.f32,semantic_id.png}
    conditions_coarse/      coarse proxy (rtcwm.data.proxy)
    cameras.json            per-frame UE camera pose + RGB intrinsics
    scene_static.json       level boxes, lane waypoints, parked vehicles (input of the coarse proxy)
    scene_frames.jsonl      per-frame protagonist bones and background walkers
    meta.json               map, seed, route, camera parameters, trajectory, QC
    prompts/NNN.txt         one caption per 124-frame window
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import PROXY_H, PROXY_W

FPS = 24
WINDOW_FRAMES, STRIDE_FRAMES = 124, 90


def n_windows(frames: int) -> int:
    return (frames - WINDOW_FRAMES) // STRIDE_FRAMES + 1


def window_frames(w: int) -> slice:
    return slice(STRIDE_FRAMES * w, STRIDE_FRAMES * w + WINDOW_FRAMES)


def read_meta(clip: Path) -> dict:
    return json.loads((Path(clip) / "meta.json").read_text())


def read_cameras(clip: Path) -> list:
    return json.loads((Path(clip) / "cameras.json").read_text())["frames"]


def read_scene_frames(clip: Path) -> list:
    with open(Path(clip) / "scene_frames.jsonl") as fh:
        return [json.loads(line) for line in fh]


def read_scene_static(clip: Path) -> dict:
    return json.loads((Path(clip) / "scene_static.json").read_text())


def read_proxy(clip: Path, i: int, kind: str = "conditions"):
    """(depth [192,336] float32, semantic [192,336] uint8) of frame i."""
    d = Path(clip) / kind
    depth = np.fromfile(d / f"{i:06d}.depth.f32", dtype="<f4").reshape(PROXY_H, PROXY_W)
    sem = np.array(Image.open(d / f"{i:06d}.semantic_id.png"))
    return depth, sem


def qc_pass(clip: Path) -> bool:
    try:
        return bool(read_meta(clip).get("qc", {}).get("pass"))
    except (OSError, ValueError):
        return False


def has_coarse_proxy(clip: Path) -> bool:
    d = Path(clip) / "conditions_coarse"
    try:
        return d.is_dir() and len(os.listdir(d)) >= 2 * read_meta(clip)["frames"]
    except (OSError, ValueError, KeyError):
        return False


def has_prompts(clip: Path) -> bool:
    try:
        nw = n_windows(read_meta(clip)["frames"])
    except (OSError, ValueError, KeyError):
        return False
    return all((Path(clip) / "prompts" / f"{w:03d}.txt").exists() for w in range(nw))


def complete(clip: Path) -> bool:
    """Rendered, passed QC and has its coarse proxy."""
    return qc_pass(clip) and has_coarse_proxy(clip)
