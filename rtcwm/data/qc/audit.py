"""Wider render audit (all from saved outputs).

  camera : single-frame pops (a step > 2.5x both neighbours), roll
  hero   : walking backwards (toe direction vs velocity), standing still, off screen, too small /
           too large, other walkers or parked cars overlapping the protagonist
  video  : black / blown-out / duplicated frames, mean luma, exposure changes (largest frame-to-frame
           luma jump, largest luma swing within 1 s -- CARLA's auto exposure adapting)
  proxy  : protagonist present in the fine proxy but missing from the coarse one, coarse-vs-fine
           depth disagreement
"""
from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np

from ..clip import FPS, n_windows, read_cameras, read_meta, read_proxy, read_scene_frames, read_scene_static, window_frames
from ..geometry import CLS_HERO, CLS_SKY, PROXY_H, PROXY_W, longest_run, project


def camera_pops(cams) -> np.ndarray:
    """Frames where the camera jumps: position or yaw step much larger than both neighbours. The
    smooth ~0.6 s swing round the protagonist at the turn-round is FollowCamera behaviour, not a pop."""
    C = np.array([c["loc"] for c in cams])
    rot = np.array([c["rot"] for c in cams])
    s = np.r_[0, np.linalg.norm(np.diff(C, axis=0), axis=1), 0]
    y = np.r_[0, np.abs((np.diff(rot[:, 1]) + 180) % 360 - 180), 0]
    pop = np.zeros(len(C), bool)
    for i in range(1, len(C)):
        pop[i] = (s[i] > 0.15 and s[i] > 2.5 * max(s[i - 1], s[i + 1], 0.02)) or \
                 (y[i] > 4 and y[i] > 2.5 * max(y[i - 1], y[i + 1], 0.3))
    return pop


def video_stats(path: Path):
    luma, blown, dup, prev = [], [], [], None
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            g = f.reformat(width=PROXY_W, height=PROXY_H, format="gray").to_ndarray().astype(np.int16)
            luma.append(float(g.mean()))
            blown.append(float((g >= 250).mean()))
            dup.append(prev is not None and float(np.abs(g - prev).mean()) < 0.05)
            prev = g
    return np.array(luma), np.array(blown), np.array(dup)


def _touching(p, z, walkers, vehicles) -> bool:
    for w in walkers:
        if math.hypot(w[0] - p[0], w[1] - p[1]) < 0.45 and abs(w[2] - z) < 1.0:
            return True
    for v in vehicles:
        yaw = math.radians(v["r"][1])
        d = p - np.array(v["c"][:2])
        lx, ly = d[0] * math.cos(yaw) + d[1] * math.sin(yaw), -d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
        if abs(lx) < v["e"][0] + 0.15 and abs(ly) < v["e"][1] + 0.15:
            return True
    return False


def audit(clip: Path) -> dict:
    clip = Path(clip)
    meta, cams, frames = read_meta(clip), read_cameras(clip), read_scene_frames(clip)
    vehicles = read_scene_static(clip)["vehicles"]
    n = min(len(cams), len(frames))
    traj = np.array(meta["hero_traj"])[:n]
    rot = np.array([c["rot"] for c in cams[:n]])
    pops = camera_pops(cams[:n])

    vel = np.zeros((n, 2))
    vel[6:-6] = (traj[12:, :2] - traj[:-12, :2]) / (12 / FPS)
    speed = np.linalg.norm(vel, axis=1)
    back = np.zeros(n, bool)
    for i in range(n):
        b = frames[i]["bones"]
        face = sum(np.array(b[f"crl_toe__{s}"][:2]) - np.array(b[f"crl_foot__{s}"][:2]) for s in "LR")
        if speed[i] > 0.5 and np.linalg.norm(face) > 1e-3:
            back[i] = face @ vel[i] / (np.linalg.norm(face) * speed[i]) < -0.3
    touch = np.array([_touching(traj[i, :2], traj[i, 2], frames[i]["walkers"], vehicles) for i in range(n)])

    hero_frac, coarse_miss, depth_err, offscreen = [], [], [], []
    for i in range(0, n, 4):
        df, sf = read_proxy(clip, i)
        dc, sc = read_proxy(clip, i, "conditions_coarse")
        hf = sf == CLS_HERO
        hero_frac.append(float(hf.mean()))
        coarse_miss.append(hf.sum() > 50 and (hf & (sc == CLS_HERO)).sum() < 0.3 * hf.sum())
        u, v, z = project(frames[i]["bones"]["crl_hips__C"], cams[i]["loc"], cams[i]["rot"])
        offscreen.append(not (z[0] > 0 and 0 <= u[0] < PROXY_W and 0 <= v[0] < PROXY_H))
        bg = (df > 0) & (dc > 0) & (sf != CLS_SKY) & (sc != CLS_SKY)
        depth_err.append(float(np.median(np.abs(np.log(dc[bg] / df[bg])))) if bg.sum() > 100 else 0.0)
    hero_frac, depth_err = np.array(hero_frac), np.array(depth_err)
    luma, blown, dup = video_stats(clip / "rgb.mp4")
    m = min(len(luma), n)

    windows = []
    for w in range(n_windows(n)):
        sl = window_frames(w)
        a, b = sl.start, sl.stop
        a4, b4 = a // 4, (b + 3) // 4
        windows.append(dict(
            w=w, camera_pops=int(pops[a:b].sum()), max_roll=float(np.abs(rot[a:b, 2]).max()),
            backwards_s=longest_run(back[a:b]) / FPS, still_s=longest_run(speed[a:b] < 0.2) / FPS,
            touch_s=longest_run(touch[a:b]) / FPS, offscreen_s=longest_run(offscreen[a4:b4]) * 4 / FPS,
            hero_frac_min=float(hero_frac[a4:b4].min()), hero_frac_max=float(hero_frac[a4:b4].max()),
            coarse_hero_miss=int(sum(coarse_miss[a4:b4])), depth_logerr_med=float(np.median(depth_err[a4:b4])),
            luma=float(luma[a:min(b, m)].mean()), luma_min=float(luma[a:min(b, m)].min()),
            blown_max=float(blown[a:min(b, m)].max()), dup_frames=int(dup[a:min(b, m)].sum()),
            luma_jump_max=float(np.abs(np.diff(luma[a:min(b, m)])).max()),
            luma_swing_1s=float(max(np.ptp(luma[i:i + FPS]) for i in range(a, max(a + 1, min(b, m) - FPS))))))
    return dict(clip=clip.name, map=meta["map"], weather=meta.get("weather"), frames=n, video_frames=int(len(luma)),
                windows=windows)
