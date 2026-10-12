"""Third-person camera: open-code-world-model's FollowCamera, kept out of solid scenery.

FollowCamera (critically damped boom + stiffer aim, subject held at a fixed screen position) is the
camera that produces test-time proxies, so training clips use it with parameters drawn from the
ranges open-code calibrated against the released examples. open-code has no walls; here the boom is
shortened to stay clear of them:

  * rays from the protagonist's head, hips and knees to the camera and to points 0.3 m around it
    (one head-to-camera ray passed over a low wall and left the camera inside its cap stone)
  * thin things -- posts, signs, passers-by -- only count right at the camera (they used to yank it
    in by 1-3 m within one frame, ~450 such pops in dataset_v1; now it neither pops nor passes
    through them)
  * the boom length is rate-limited: in at <= 4 m/s, out at <= 1.5 m/s
  * a shortened boom re-aims at the point FollowCamera was looking at (keeping the long boom's
    angles put the protagonist out of frame at wall corners during the turn-round)
"""
from __future__ import annotations

import math

import carla

from ... import paths
from ..geometry import carla_to_renderer, renderer_to_carla

HIP_TO_FEET = 0.93                     # walker actor origin is at the hips
EYE_ABOVE_HIP = 0.75
RAY_ORIGINS = (0.75, 0.0, -0.45)       # head, hips, knees (relative to the actor origin)
RAY_SPREAD = 0.3
BOOM_IN_SPEED, BOOM_OUT_SPEED = 4.0, 1.5
MIN_BOOM_M = 0.8
YAW_TAU = 0.25


def sample_params(rng) -> dict:
    return dict(distance=rng.uniform(1.8, 4.0), height=rng.uniform(1.3, 2.5), aim_height=0.875,
                aim_screen_x=rng.uniform(0.36, 0.58), aim_screen_y=rng.uniform(0.51, 0.57),
                focal_px=206.6, boom_tau=0.70, zeta=0.50, aim_tau=0.15)


class FollowRig:
    def __init__(self, world, hero, params: dict):
        paths.use_ocwm()
        from compiler.follow import FollowCamera
        self.world, self.hero, self.params = world, hero, params
        self.follow = FollowCamera(**params)
        self.yaw = None                # smoothed walking direction
        self.k = None                  # current boom fraction (1 = full length)
        L = carla.CityObjectLabel
        self.thin = {getattr(L, n) for n in ("Poles", "TrafficSigns", "TrafficLight", "Pedestrians", "Rider", "Other")
                     if hasattr(L, n)}

    def _heading(self, ht, dt):
        v = self.hero.get_velocity()
        head = ht.rotation.yaw if math.hypot(v.x, v.y) < 0.3 else math.degrees(math.atan2(v.y, v.x))
        if self.yaw is None:
            self.yaw = head
        self.yaw += ((head - self.yaw + 180.0) % 360.0 - 180.0) * (1.0 - math.exp(-dt / YAW_TAU))
        return self.yaw

    def _clear_fraction(self, ht, want, d, dist):
        side = carla.Location(x=-d.y / dist, y=d.x / dist, z=0.0)
        offs = [carla.Location()] + [side * s for s in (RAY_SPREAD, -RAY_SPREAD)] + \
               [carla.Location(z=s) for s in (RAY_SPREAD, -RAY_SPREAD)]
        k = 1.0
        for oz in RAY_ORIGINS:
            o = ht.location + carla.Location(z=oz)
            for off in offs:
                tgt = want + off
                seg = o.distance(tgt)
                for h in self.world.cast_ray(o, tgt):
                    t = h.location.distance(o)
                    if t > 0.4 and (h.label not in self.thin or t > seg - 0.6):
                        k = min(k, max(0.25, (t - 0.35) / seg))
        return max(k, min(1.0, MIN_BOOM_M / dist))

    def pose(self, dt: float) -> carla.Transform:
        ht = self.hero.get_transform()
        feet = ht.location + carla.Location(z=-HIP_TO_FEET)
        c = self.follow.update(carla_to_renderer([feet.x, feet.y, feet.z]), self._heading(ht, dt), dt)
        p0 = carla.Location(*renderer_to_carla(c.position))
        want = p0
        eye = ht.location + carla.Location(z=EYE_ABOVE_HIP)
        d = want - eye
        dist = math.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2)
        if dist > 0.5:
            k = self._clear_fraction(ht, want, d, dist)
            if self.k is None:         # first frame: no history to rate-limit against
                self.k = k
            if k < self.k:
                self.k = max(k, self.k - BOOM_IN_SPEED * dt / dist)
            else:
                self.k = min(k, self.k + BOOM_OUT_SPEED * dt / dist)
            if self.k < 1.0:
                want = eye + carla.Location(x=d.x * self.k, y=d.y * self.k, z=d.z * self.k)
        pitch, yaw = c.pitch_deg, c.yaw_deg
        if want.distance(p0) > 0.01:
            pr, yr = math.radians(pitch), math.radians(yaw)
            f0 = carla.Location(x=math.cos(pr) * math.cos(yr), y=math.cos(pr) * math.sin(yr), z=math.sin(pr))
            aim = feet + carla.Location(z=self.params["aim_height"])
            v = (p0 + f0 * p0.distance(aim)) - want
            yaw = math.degrees(math.atan2(v.y, v.x))
            pitch = math.degrees(math.atan2(v.z, math.hypot(v.x, v.y)))
        return carla.Transform(want, carla.Rotation(pitch=pitch, yaw=yaw))
