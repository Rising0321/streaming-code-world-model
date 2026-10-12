"""Interpenetration in a rendered clip, from the fine depth/semantic, camera poses and bones.

  buried   : the protagonist's feet are below the visible ground -- at the projected toe/foot
             bones the depth buffer shows a surface clearly in front of the bone (Town04).
  inside   : the protagonist's hips are over the footprint of a hedge, planter, wall, fence or
             static prop that spans its shins.
  pierced  : a pole or other thin solid runs through the body: in the middle 60% of the
             protagonist's box, a column that is mostly non-protagonist surface at its own depth.
  occluded : something (typically a wall the camera is inside) covers > 30% of the protagonist's
             projected bone box with surfaces well in front of it and raised above its feet
             (the ground in front of the feet and walls beside the path do not count).

A window is bad if buried / inside / occluded holds for > 0.5 s in a row, or pierced for >= 3 frames
(walking through a post takes ~0.3 s).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..clip import FPS, n_windows, read_cameras, read_proxy, read_scene_frames, read_scene_static, window_frames
from ..geometry import (CLS_HERO, CLS_SKY, FEET_BONES, PROXY_CX, PROXY_CY, PROXY_F, PROXY_H, PROXY_W, camera_basis,
                        longest_run)

RUN = int(0.5 * FPS)
PIERCED_RUN = 3
SOLID_LABELS = ("Walls", "Fences", "GuardRail", "Static")   # + vegetation under 2.4 m (hedges, planters)


def _solid_boxes(clip: Path):
    """Low solid things a walker must not stand in. Tree boxes are excluded (their canopy footprint
    covers sidewalks), and so are buildings (their boxes include awnings)."""
    try:
        boxes = read_scene_static(clip)["boxes"]
    except OSError:
        return []
    out = []
    for b in boxes:
        e = [abs(x) for x in b["e"]]
        if (b["label"] in SOLID_LABELS or (b["label"] == "Vegetation" and e[2] < 1.2)) and max(e[0], e[1]) > 0.2:
            out.append((np.array(b["c"]), np.array(e), math.radians(b["r"][1])))
    return out


def _buried(d, s, bones, C, fwd, right, up) -> bool:
    sunk = 0
    for b in FEET_BONES:
        q = np.array(bones[b]) - C
        z = q @ fwd
        if z < 0.3:
            continue
        iu, iv = int(PROXY_CX + PROXY_F * (q @ right) / z), int(PROXY_CY - PROXY_F * (q @ up) / z)
        if not (1 <= iu < PROXY_W - 1 and 1 <= iv < PROXY_H - 1):
            continue
        win_d, win_s = d[iv - 1:iv + 2, iu - 1:iu + 2], s[iv - 1:iv + 2, iu - 1:iu + 2]
        if not (win_s == CLS_HERO).any() and np.median(win_d) < z - 0.15:
            sunk += 1
    return sunk >= 3


def _inside(bones, foot_z, solid) -> bool:
    hp = np.array(bones["crl_hips__C"])
    for bc, be, yaw in solid:
        if not (bc[2] - be[2] < foot_z + 0.5 and bc[2] + be[2] > foot_z + 0.3):
            continue
        dx, dy = hp[0] - bc[0], hp[1] - bc[1]
        u, v = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
        if abs(u) < be[0] - 0.1 and abs(v) < be[1] - 0.1:
            return True
    return False


def _box_checks(d, s, bones, C, fwd, right, up, foot_z, xn, yn):
    """(occluded fraction of the protagonist's box, pole through the body)."""
    hd = (np.array(bones["crl_hips__C"]) - C) @ fwd
    pts = np.array(list(bones.values())) - C
    zs = pts @ fwd
    if not (zs > 0.3).all():
        return 0.0, False
    us, vs = PROXY_CX + PROXY_F * (pts @ right) / zs, PROXY_CY - PROXY_F * (pts @ up) / zs
    u0, u1 = int(max(0, us.min() - 3)), int(min(PROXY_W, us.max() + 4))
    v0, v1 = int(max(0, vs.min() - 3)), int(min(PROXY_H, vs.max() + 4))
    if u1 - u0 < 3 or v1 - v0 < 3:
        return 0.0, False
    bd, bs = d[v0:v1, u0:u1], s[v0:v1, u0:u1]
    height = C[2] + bd * (fwd[2] + xn[v0:v1, u0:u1] * right[2] + yn[v0:v1, u0:u1] * up[2]) - foot_z
    occ = float(((bd > 0) & (bd < 0.75 * hd) & (height > 0.3) & (bs != CLS_HERO) & (bs != CLS_SKY)).mean())
    hz = np.median(bd[bs == CLS_HERO]) if (bs == CLS_HERO).sum() > 20 else hd
    same = (np.abs(bd - hz) < 0.3) & (bs != CLS_HERO) & (bs != CLS_SKY) & (height > 0.25)
    w0 = int(0.2 * (u1 - u0))
    cols = same[:, w0:(u1 - u0) - w0].mean(0) if (u1 - u0) - 2 * w0 > 0 else np.zeros(1)
    return occ, bool(cols.max() > 0.5)


def scan(clip: Path) -> dict:
    clip = Path(clip)
    cams, frames = read_cameras(clip), read_scene_frames(clip)
    n = min(len(cams), len(frames))
    vv, uu = np.mgrid[0:PROXY_H, 0:PROXY_W]
    xn, yn = (uu + 0.5 - PROXY_CX) / PROXY_F, -(vv + 0.5 - PROXY_CY) / PROXY_F
    solid = _solid_boxes(clip)
    buried, inside, occluded, occ_frac, pierced = [], [], [], [], []
    for i in range(n):
        d, s = read_proxy(clip, i)
        C = np.array(cams[i]["loc"])
        fwd, right, up = camera_basis(*cams[i]["rot"][:2])
        bones = frames[i]["bones"]
        foot_z = min(bones[b][2] for b in FEET_BONES)
        buried.append(_buried(d, s, bones, C, fwd, right, up))
        inside.append(_inside(bones, foot_z, solid))
        frac, pole = _box_checks(d, s, bones, C, fwd, right, up, foot_z, xn, yn)
        occ_frac.append(round(frac, 3))
        occluded.append(frac > 0.3)
        pierced.append(pole)
    windows = []
    for w in range(n_windows(n)):
        sl = window_frames(w)
        r = dict(w=w, buried_run=longest_run(buried[sl]), occluded_run=longest_run(occluded[sl]),
                 inside_run=longest_run(inside[sl]), pierced_run=longest_run(pierced[sl]))
        r["bad"] = r["pierced_run"] >= PIERCED_RUN or max(r["buried_run"], r["occluded_run"], r["inside_run"]) > RUN
        windows.append(r)
    return dict(clip=clip.name, frames=n,
                max_buried_run=longest_run(buried), max_occluded_run=longest_run(occluded),
                max_inside_run=longest_run(inside), max_pierced_run=longest_run(pierced),
                bad_windows=[x["w"] for x in windows if x["bad"]], windows=windows,
                buried=[int(b) for b in buried], inside=[int(b) for b in inside],
                pierced=[int(b) for b in pierced], occ_frac=occ_frac)
