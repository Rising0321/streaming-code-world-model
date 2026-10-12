"""Coarse CWM proxy of a rendered clip, compiled from its scene description.

The fine proxy (conditions/) is CARLA's own depth + segmentation of the real meshes. At test time a
coding agent only has coarse geometry, which is what CWM's proxy is. This rebuilds that with
open-code-world-model's primitives and ray tracer, from the exact camera of every RGB frame:

    buildings, walls, props -> their CARLA level boxes, trimmed clear of the walk and camera path
                               (Town10HD building boxes include awnings over the sidewalk)
    poles, lights, signs    -> a capsule (+ a small head box); their boxes include the lamp arm
    vegetation              -> open-code lobed clumps, split along long boxes, lobes kept inside the
                               box (its Gaussian scatter once hung a lobe over the camera)
    roads                   -> flat strips from driving-lane waypoints; ground -> one plane
    backdrop                -> a ring of distant boxes (open-code: 320-900 m), so the skyline is geometry
    protagonist             -> open-code's generic Human rig (class 8): position, heading and a
                               procedural gait only -- no body shape or clothing
    other walkers           -> the same rig (class 0); vehicles -> wireframe boxes (class 10)

    python -m rtcwm.data.proxy [--workers 32] [--out conditions_coarse] CLIP_DIR ...
"""
from __future__ import annotations

import argparse
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import paths
from .clip import FPS, read_cameras, read_meta, read_scene_frames, read_scene_static
from .geometry import CLS_HERO, CLS_OTHER, CLS_ROAD, CLS_VEGETATION, CLS_VEHICLE, CLS_WATER, PROXY_F
from .geometry import carla_to_renderer as R
from .geometry import renderer_axes as axes

paths.use_ocwm()
from compiler import writer                                                     # noqa: E402
from compiler.render import Box, Camera, Capsule, Plane, render_frame, vegetation_clump, wireframe_box  # noqa: E402
from compiler.rig import Human, idle_pose, walk_pose                            # noqa: E402

CULL_M = 200.0
HERO_SCALE = 1.75 / 1.60          # open-code rig is 1.60 m; CARLA adults are ~1.75 m
WALKER_RADIUS_M = 60.0
PATH_CLEARANCE_M = 0.4


def covers_road(b, lanes_xy) -> bool:
    """A box whose footprint contains a driving-lane waypoint: whole-block building meshes have
    boxes that span the street, and a coarse box there would wall the road off."""
    c, e, yaw = b["c"], b["e"], math.radians(b["r"][1])
    d = lanes_xy - np.array(c[:2])
    u = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
    v = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
    return bool(np.any((np.abs(u) < e[0] - 0.5) & (np.abs(v) < e[1] - 0.5)))


def clear_of_path(c, e, yaw_deg, pts, margin=PATH_CLEARANCE_M):
    """Trim a solid box so no path point lies within `margin` of it: each offending point pushes the
    nearer of the box's in-plane faces back past it. None if nothing is left."""
    c, e = np.array(c, float), np.array(e, float)
    yaw = math.radians(yaw_deg)
    ax_u, ax_v = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    lo, hi = -e[:2].copy(), e[:2].copy()
    for p in pts:
        if not (c[2] - e[2] - 0.5 <= p[2] <= c[2] + e[2] + 0.5):
            continue
        d = np.array(p[:2]) - c[:2]
        q = np.array([d @ ax_u, d @ ax_v])
        if not np.all((q > lo - margin) & (q < hi + margin)):
            continue
        moves = [(hi[k] - (q[k] - margin), k, "hi") for k in range(2)] + \
                [((q[k] + margin) - lo[k], k, "lo") for k in range(2)]
        _, k, side = min(moves)                     # cheapest single-face move (box area removed)
        if side == "hi":
            hi[k] = q[k] - margin
        else:
            lo[k] = q[k] + margin
        if hi[k] - lo[k] < 0.2:
            return None
    mid = (lo + hi) / 2
    c2, e2 = c.copy(), e.copy()
    c2[:2] = c[:2] + mid[0] * ax_u + mid[1] * ax_v
    e2[:2] = (hi - lo) / 2
    return c2, e2


def vegetation_prims(c, e, yaw_deg):
    """Lobed clumps for one vegetation box; trees (>= 2.4 m tall) get a canopy on a trunk."""
    yaw = math.radians(yaw_deg)
    long_x = e[0] >= e[1]
    major, minor = (e[0], e[1]) if long_x else (e[1], e[0])
    n = max(1, int(round(major / max(minor, 0.5))))
    ux = (math.cos(yaw), math.sin(yaw)) if long_x else (-math.sin(yaw), math.cos(yaw))
    half = max(major / n, minor)
    top, bot = c[2] + e[2], c[2] - e[2]

    def inside(lobes, zmin):
        kept = []
        for s in lobes:
            p = np.array([s.centre[2], s.centre[0], -s.centre[1]])        # renderer -> CARLA
            d = p[:2] - np.array(c[:2])
            u = d[0] * math.cos(yaw) + d[1] * math.sin(yaw)
            v = -d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
            if abs(u) <= e[0] and abs(v) <= e[1] and zmin <= p[2] <= top:
                kept.append(s)
        return kept

    prims = []
    for k in range(n):
        t = -major + (2 * k + 1) * major / n
        pc = [c[0] + t * ux[0], c[1] + t * ux[1], c[2]]
        seed = int(abs(pc[0] * 7.3 + pc[1] * 3.1)) % 10000
        if e[2] < 1.2:
            prims += inside(vegetation_clump(R(pc), 0.45 * half, aspect=e[2] / max(half, 0.1), seed=seed), bot)
        else:
            ch = 0.35 * e[2]
            cc = [pc[0], pc[1], top - ch]
            rad = 0.45 * half
            prims += inside(vegetation_clump(R(cc), rad, aspect=ch / max(rad, 0.1), seed=seed), max(top - 2 * ch, bot + 2.0))
            prims.append(Capsule(R([pc[0], pc[1], bot]), R(cc), 0.18, CLS_VEGETATION))
    return prims


def static_prims(scene, centre, ground_z, keep_clear=()):
    prims = []
    lanes_xy = np.array([[p[0], p[1]] for p in scene["lanes"]]) if scene["lanes"] else np.zeros((0, 2))
    for b in scene["boxes"]:
        c, e = b["c"], b["e"]
        if math.hypot(c[0] - centre[0], c[1] - centre[1]) > CULL_M or max(e) > 80.0:
            continue                                            # whole-map landscape boxes swallow the camera
        # coarse means coarse: no grass clumps or small props a coding agent would never place
        if (b["cls"] == CLS_VEGETATION and e[2] < 0.4) or (b["cls"] == CLS_OTHER and max(e) < 0.25):
            continue
        if b["label"] in ("Static", "Dynamic") and max(e) < 1.0:
            continue
        if b["label"] in ("Buildings", "Walls") and len(lanes_xy) and covers_road(b, lanes_xy):
            continue
        e = [abs(x) for x in e]
        ax = axes(b["r"][1])
        if b["label"] in ("Poles", "TrafficLight", "TrafficSigns"):
            bot, top = c[2] - e[2], c[2] + e[2]
            prims.append(Capsule(R([c[0], c[1], bot]), R([c[0], c[1], top]), 0.12, CLS_OTHER))
            if b["label"] != "Poles":
                prims.append(Box(R([c[0], c[1], top - 0.4]), np.array([0.3, 0.3, 0.4]), ax, class_id=CLS_OTHER))
        elif b["cls"] == CLS_VEGETATION:
            prims += vegetation_prims(c, e, b["r"][1])
        elif b["cls"] == CLS_WATER:
            prims.append(Box(R(c), np.array([e[0], e[1], max(e[2], 0.02)]), ax, class_id=CLS_WATER))
        else:
            if len(keep_clear):
                cut = clear_of_path(c, e, b["r"][1], keep_clear)
                if cut is None:
                    continue
                c, e = list(cut[0]), list(cut[1])
            prims.append(Box(R(c), np.array(e), ax, class_id=CLS_OTHER))
    for x, y, z, yaw, width in scene["lanes"]:
        if math.hypot(x - centre[0], y - centre[1]) <= CULL_M and abs(z - ground_z) <= 1.5:   # not another level
            prims.append(Box(R([x, y, ground_z + 0.01]), np.array([1.05, width / 2.0, 0.01]), axes(yaw), class_id=CLS_ROAD))
    rng = np.random.default_rng(int(abs(centre[0] * 13 + centre[1] * 7)) % 2**31)
    for k in range(11):                                         # distant backdrop ring
        ang = 2 * math.pi * k / 11
        dist = float(rng.uniform(320.0, 900.0))
        size = [float(rng.uniform(60.0, 160.0)), float(rng.uniform(60.0, 160.0)), float(rng.uniform(45.0, 130.0))]
        bx = [centre[0] + dist * math.cos(ang), centre[1] + dist * math.sin(ang)]
        prims.append(Box(R([bx[0], bx[1], ground_z + size[2] / 2]), np.array([s / 2 for s in size]),
                         axes(math.degrees(ang)), class_id=CLS_OTHER))
    for v in scene["vehicles"]:
        if math.hypot(v["c"][0] - centre[0], v["c"][1] - centre[1]) <= CULL_M:
            prims += wireframe_box(R(v["c"]), np.array(v["e"]), axes(v["r"][1]), class_id=CLS_VEHICLE)
    return prims


def protagonist(x, y, ground_z, yaw, phase, moving):
    pose = walk_pose(phase) if moving else idle_pose(phase)
    return Human(position=R([x, y, ground_z]), facing_deg=yaw, pose=pose, scale=HERO_SCALE).primitives()


def walkers(poses, ground_z, t):
    prims = []
    for k, (x, y, _z, yaw) in enumerate(poses):
        h = Human(position=R([x, y, ground_z]), facing_deg=yaw, pose=walk_pose((t / 0.95 + 0.37 * k) % 1.0))
        for p in h.primitives():
            p.class_id = CLS_OTHER                              # other people are class 0 in CWM
            prims.append(p)
    return prims


def headings(xy):
    """Protagonist heading from the smoothed motion (degrees, CARLA yaw)."""
    yaw, out = None, []
    for i in range(len(xy)):
        d = xy[min(len(xy) - 1, i + 4)] - xy[max(0, i - 4)]
        if np.linalg.norm(d) > 0.15:
            target = math.degrees(math.atan2(d[1], d[0]))
            yaw = target if yaw is None else yaw + ((target - yaw + 180) % 360 - 180) * 0.35
        out.append(yaw if yaw is not None else 0.0)
    return out


def _render(job):
    i, cam, walkers_i, static, ground_z, hero = job
    prims = list(static)
    prims.append(Plane(np.array([0.0, -ground_z, 0.0]), np.array([0.0, -1.0, 0.0]), class_id=CLS_OTHER))
    prims += protagonist(*hero)
    near = [w for w in walkers_i if math.hypot(w[0] - cam["loc"][0], w[1] - cam["loc"][1]) < WALKER_RADIUS_M]
    prims += walkers(near, ground_z, i / FPS)
    c = Camera(focal_px=PROXY_F, position=R(cam["loc"]), pitch_deg=cam["rot"][0], yaw_deg=cam["rot"][1], roll_deg=0.0)
    z, ids = render_frame(c, prims)
    return i, z, ids


def compile_clip(clip: Path, out_name: str = "conditions_coarse", workers: int = 32) -> int:
    clip = Path(clip)
    scene, cams, frames = read_scene_static(clip), read_cameras(clip), read_scene_frames(clip)
    traj = np.array(read_meta(clip)["hero_traj"])
    feet0 = [v[2] for k, v in frames[0]["bones"].items() if "foot" in k or "toe" in k]
    path_pts = [list(p) for p in traj[::4]] + [c["loc"] for c in cams[::4]]
    static = static_prims(scene, traj[:, :2].mean(0), (min(feet0) - 0.03) if feet0 else traj[0, 2] - 0.95,
                          keep_clear=path_pts)
    xy = traj[:, :2]
    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    dist = np.cumsum(step)
    speed = np.convolve(step, np.ones(9) / 9, mode="same") * FPS
    yaws = headings(xy)
    jobs = []
    for i, (cam, sf) in enumerate(zip(cams, frames)):
        feet = [v[2] for k, v in sf["bones"].items() if "foot" in k or "toe" in k]
        ground_z = min(feet) - 0.03 if feet else traj[i, 2] - 0.95
        moving = speed[i] > 0.25
        phase = (dist[i] / 1.5) % 1.0 if moving else (i / 48.0) % 1.0     # 1.5 m per stride cycle
        jobs.append((i, cam, sf["walkers"], static, ground_z, (traj[i, 0], traj[i, 1], ground_z, yaws[i], phase, moving)))
    with Pool(workers) as pool:
        for i, z, ids in pool.imap_unordered(_render, jobs, chunksize=4):
            writer.write_frame(clip / out_name, i, z, ids)
    return len(static)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--out", default="conditions_coarse", help="output directory name inside each clip")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    for clip in a.clips:
        n = compile_clip(clip, a.out, a.workers)
        print(clip.name, "coarse proxy:", n, "static primitives", flush=True)


if __name__ == "__main__":
    main()
