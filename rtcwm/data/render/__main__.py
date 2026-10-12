"""python -m rtcwm.data.render --port P --map MAP --seconds S --seed N --out OUT"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import av
import carla
import numpy as np

from ... import paths
from . import camera, scene
from .config import FPS, HFOV, LEG_PER_SECOND, RGB_H, RGB_W, WALK_SPEED, WARMUP_S, WEATHERS
from .qc import RenderQC, add_clipping
from .route import RoutePlanner, Steerer
from .sensors import SensorRig, decode, leg_clip_fraction


def parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--seconds", type=float, default=38.916666666666664)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weather", default=None)
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=30)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse()
    a.seed += int(os.environ.get("CARLA_SEED_OFFSET", "0"))     # set by run_clip.sh on a QC retry
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "conditions").mkdir(parents=True, exist_ok=True)
    paths.use_ocwm()
    from compiler import writer
    from compiler.render import pool

    client = carla.Client(a.host, a.port)
    client.set_timeout(300.0)
    world = client.get_world()
    if not world.get_map().name.endswith(a.map):
        world = client.load_world(a.map)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)
    world.set_pedestrians_seed(a.seed)
    weather = a.weather or rng.choice(WEATHERS)
    world.set_weather(getattr(carla.WeatherParameters, weather))

    bp = world.get_blueprint_library()
    traffic, rig, hero_ctrl = scene.Traffic(), None, None
    try:
        traffic = scene.spawn_traffic(world, bp, rng, a.vehicles, a.walkers)
        leg = LEG_PER_SECOND * a.seconds
        hero, route = RoutePlanner(world, rng).plan(leg, scene.adult_walkers(bp))
        traffic.actors.append(hero)
        print(f"route: {route.kind}, {len(route.targets)} targets, {len(route.hidden_ids)} posts/trunks hidden", flush=True)
        hero_ctrl = world.spawn_actor(bp.find("controller.ai.walker"), carla.Transform(), hero)
        traffic.controllers.append(hero_ctrl)
        world.tick()
        hero_ctrl.start()
        hero_ctrl.set_max_speed(WALK_SPEED)
        steer = Steerer(hero_ctrl, hero, route.targets)

        cam_params = camera.sample_params(rng)
        follow = camera.FollowRig(world, hero, cam_params)
        rig = SensorRig(world, bp, follow.pose(1.0 / FPS))
        for _ in range(int(WARMUP_S * FPS)):                # let the boom and the walkers settle
            rig.place(follow.pose(1.0 / FPS))
            rig.grab(world.tick())

        n_frames = int(round(a.seconds * FPS))
        container = av.open(str(out / "rgb.mp4"), "w")
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = RGB_W, RGB_H, "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        qc, cams, traj, frames = RenderQC(), [], [], []
        t0 = time.time()
        for i in range(n_frames):
            rig.place(follow.pose(1.0 / FPS))
            imgs = rig.grab(world.tick())
            hl = hero.get_location()
            steer.step(i)
            f = decode(imgs, hl, rig.rgb)
            for p in stream.encode(av.VideoFrame.from_ndarray(f.rgb, format="rgb24")):
                container.mux(p)
            zp, ip = pool(f.z, f.ids)
            writer.write_frame(out / "conditions", i, zp, ip)
            qc.add(hero_frac=float((ip == 8).mean()), near_frac=float(((zp > 0) & (zp < 0.6)).mean()),
                   leg_clip=leg_clip_fraction(f), luma=float(f.rgb.mean()) if i % FPS == 0 else None)
            ct = rig.rgb.get_transform()
            cams.append(dict(loc=[ct.location.x, ct.location.y, ct.location.z],
                             rot=[ct.rotation.pitch, ct.rotation.yaw, ct.rotation.roll]))
            traj.append([hl.x, hl.y, hl.z])
            frames.append(scene.frame_record(hero, traffic.walkers))
            if i % 240 == 0:
                print(f"frame {i}/{n_frames}  {(i + 1) / (time.time() - t0):.1f} fps", flush=True)
        for p in stream.encode():
            container.mux(p)
        container.close()

        verdict = qc.summary(traj, leg, WALK_SPEED, a.seconds)
        centre = (float(np.mean(np.array(traj)[:, 0])), float(np.mean(np.array(traj)[:, 1])))
        (out / "scene_static.json").write_text(json.dumps(scene.scene_static(world, centre, route.hidden_boxes,
                                                                             traffic.vehicles)))
        with open(out / "scene_frames.jsonl", "w") as fh:
            for sf in frames:
                fh.write(json.dumps(sf) + "\n")
        fx = (RGB_W / 2) / math.tan(math.radians(HFOV) / 2)
        (out / "cameras.json").write_text(json.dumps(dict(
            convention="UE: x forward, y right, z up, metres; rot = pitch, yaw, roll degrees",
            intrinsics_rgb=dict(w=RGB_W, h=RGB_H, fx=fx, fy=fx, cx=RGB_W / 2, cy=RGB_H / 2, hfov_deg=HFOV),
            frames=cams)))
        verdict = add_clipping(verdict, out)
        print("qc", verdict, flush=True)
        hero_frac = np.array(qc.hero_frac)
        (out / "meta.json").write_text(json.dumps(dict(
            map=a.map, weather=weather, route="outback", seed=a.seed, frames=n_frames, fps=FPS,
            route_points=[[p.x, p.y, p.z] for p in route.targets], hero_traj=traj, plan="out-back-out", leg_m=leg,
            route_kind=route.kind, camera="open-code FollowCamera", cam_params=cam_params,
            start=[route.start.x, route.start.y, route.start.z],
            hero_visible_frac_median=float(np.median(hero_frac)), hero_visible_frac_min=float(np.min(hero_frac)),
            render_fps=n_frames / (time.time() - t0), qc=verdict), indent=1))
        print("done", out, f"{n_frames / (time.time() - t0):.1f} fps")
        if not verdict["pass"]:
            sys.exit(3)
    finally:
        if rig is not None:
            rig.stop()
        for c in traffic.controllers:
            try:
                c.stop()
            except RuntimeError:
                pass
        sensors = rig.sensors if rig is not None else []
        client.apply_batch([carla.command.DestroyActor(x) for x in sensors + traffic.controllers + traffic.actors])
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
