"""RGB / depth / instance-segmentation cameras on one pose, and decoding their frames."""
from __future__ import annotations

import math
import queue
from dataclasses import dataclass

import numpy as np

from .config import HFOV, RGB_H, RGB_W, SENSOR_H, SENSOR_W, TAG2CWM, TAG_PEDESTRIAN, TAGS_SOLID

RGB, DEPTH, SEG = "sensor.camera.rgb", "sensor.camera.depth", "sensor.camera.instance_segmentation"


class SensorRig:
    def __init__(self, world, bp, initial_tf):
        self.sensors, self.queues = [], {}
        self.rgb = self._add(world, bp, initial_tf, RGB, RGB_W, RGB_H, motion_blur_intensity=0.0)
        self._add(world, bp, initial_tf, DEPTH, SENSOR_W, SENSOR_H)
        self._add(world, bp, initial_tf, SEG, SENSOR_W, SENSOR_H)

    def _add(self, world, bp, tf, kind, w, h, **attrs):
        b = bp.find(kind)
        b.set_attribute("image_size_x", str(w))
        b.set_attribute("image_size_y", str(h))
        b.set_attribute("fov", f"{HFOV:.4f}")
        for k, v in attrs.items():
            if b.has_attribute(k):
                b.set_attribute(k, str(v))
        s = world.spawn_actor(b, tf)
        q = queue.Queue()
        s.listen(q.put)
        self.sensors.append(s)
        self.queues[kind] = q
        return s

    def place(self, tf):
        for s in self.sensors:
            s.set_transform(tf)

    def grab(self, frame: int, timeout: float = 120.0) -> dict:
        got = {}
        for kind, q in self.queues.items():
            while True:
                img = q.get(timeout=timeout)
                if img.frame == frame:
                    got[kind] = img
                    break
        return got

    def stop(self):
        for s in self.sensors:
            s.stop()


@dataclass
class Frame:
    rgb: np.ndarray            # [H,W,3] uint8
    z: np.ndarray              # [SENSOR_H,SENSOR_W] planar depth (m), sky -> 0, clipped at 0.3
    ids: np.ndarray            # CWM semantic ids at sensor resolution
    tag: np.ndarray            # raw CARLA tags
    hero: np.ndarray           # protagonist mask


def decode(imgs: dict, hero_loc, rgb_sensor) -> Frame:
    bgra = np.frombuffer(imgs[RGB].raw_data, np.uint8).reshape(RGB_H, RGB_W, 4)
    d = np.frombuffer(imgs[DEPTH].raw_data, np.uint8).reshape(SENSOR_H, SENSOR_W, 4).astype(np.float64)
    z = (d[..., 2] + d[..., 1] * 256.0 + d[..., 0] * 65536.0) / (256.0 ** 3 - 1) * 1000.0   # 1000 m far
    s = np.frombuffer(imgs[SEG].raw_data, np.uint8).reshape(SENSOR_H, SENSOR_W, 4)
    tag = s[..., 2]
    inst = s[..., 1].astype(np.int32) + 256 * s[..., 0].astype(np.int32)
    ids = TAG2CWM[tag].copy()
    hero = hero_mask(tag, inst, hero_loc, np.array(rgb_sensor.get_transform().get_inverse_matrix()))
    ids[hero] = 8
    sky = (ids == 1) | (z >= 999.0)
    ids[sky] = 1
    z = np.where(sky, 0.0, np.clip(z, 0.3, None)).astype(np.float32)
    return Frame(np.ascontiguousarray(bgra[..., 2::-1]), z, ids, tag, hero)


def hero_mask(tag, inst, hero_loc, w2c) -> np.ndarray:
    """The instance channel is an internal object id, not the actor id: project the protagonist's
    body into the image and take the pedestrian instance found there (majority vote)."""
    fpx = (SENSOR_W / 2) / math.tan(math.radians(HFOV) / 2)
    votes = {}
    for dz in (-0.4, 0.0, 0.4):
        pc = w2c @ np.array([hero_loc.x, hero_loc.y, hero_loc.z + dz, 1.0])
        if pc[0] <= 0.3:
            continue
        u = int(SENSOR_W / 2 + fpx * pc[1] / pc[0])
        v = int(SENSOR_H / 2 - fpx * pc[2] / pc[0])
        y0, y1, x0, x1 = max(v - 6, 0), min(v + 7, SENSOR_H), max(u - 6, 0), min(u + 7, SENSOR_W)
        if y0 >= y1 or x0 >= x1:
            continue
        win_t, win_i = tag[y0:y1, x0:x1], inst[y0:y1, x0:x1]
        for key in win_i[win_t == TAG_PEDESTRIAN].ravel():
            votes[int(key)] = votes.get(int(key), 0) + 1
    if not votes:
        return np.zeros_like(tag, bool)
    return (tag == TAG_PEDESTRIAN) & (inst == max(votes, key=votes.get))


def leg_clip_fraction(f: Frame) -> float:
    """Fraction of the protagonist's leg box covered by solid scenery (building, wall, fence,
    vegetation, props) at its own depth. Ground classes are excluded -- they sit at foot depth."""
    ys, xs = np.nonzero(f.hero)
    if len(ys) <= 80:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    legs = slice(y0 + int(0.55 * (y1 - y0)), y1)
    hz = float(np.median(f.z[ys, xs]))
    solid = np.isin(f.tag[legs, x0:x1], TAGS_SOLID) & (np.abs(f.z[legs, x0:x1] - hz) < 0.7)
    return float(solid.mean())
