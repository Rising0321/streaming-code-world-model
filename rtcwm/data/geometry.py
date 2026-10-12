"""Conventions shared by the renderer, the coarse proxy and the QC scans.

CARLA / UE world: x forward, y right, z up (left-handed), metres; rotations are degrees.
open-code renderer world: x right, y down, z forward.
CWM proxy frames: 336x192, focal 206.6 px, semantic ids {0,1,2,4,5,8,9,10}.
"""
from __future__ import annotations

import math

import numpy as np

PROXY_W, PROXY_H = 336, 192
PROXY_F = 206.6
PROXY_CX, PROXY_CY = PROXY_W / 2, PROXY_H / 2
HFOV_DEG = math.degrees(2 * math.atan(PROXY_CX / PROXY_F))

# CWM semantic ids
CLS_OTHER, CLS_SKY, CLS_WATER, CLS_ROAD, CLS_VEGETATION, CLS_HERO, CLS_VEHICLE = 0, 1, 2, 4, 5, 8, 10

FEET_BONES = ("crl_foot__L", "crl_foot__R", "crl_toe__L", "crl_toe__R")


def carla_to_renderer(p) -> np.ndarray:
    return np.array([p[1], -p[2], p[0]], float)


def renderer_to_carla(q) -> list:
    return [float(q[2]), float(q[0]), float(-q[1])]


def renderer_axes(yaw_deg: float) -> np.ndarray:
    """Renderer-space unit axes (rows: forward, right, up) of a CARLA yaw."""
    y = math.radians(yaw_deg)
    return np.stack([carla_to_renderer([math.cos(y), math.sin(y), 0.0]),
                     carla_to_renderer([-math.sin(y), math.cos(y), 0.0]),
                     carla_to_renderer([0.0, 0.0, 1.0])])


def camera_basis(pitch_deg: float, yaw_deg: float):
    """UE camera axes (forward, right, up) in world coordinates."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([-math.sin(y), math.cos(y), 0.0])
    up = np.cross(fwd, right)            # UE is left-handed: this is +up
    return fwd, right, up


def project(points, cam_loc, cam_rot):
    """World points [N,3] -> proxy pixel (u, v) and planar depth z for a UE camera pose."""
    fwd, right, up = camera_basis(*cam_rot[:2])
    q = np.atleast_2d(np.asarray(points, float)) - np.asarray(cam_loc, float)
    z = q @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        u = PROXY_CX + PROXY_F * (q @ right) / z
        v = PROXY_CY - PROXY_F * (q @ up) / z
    return u, v, z


def box_local(xy, centre, yaw_deg: float):
    """2D offsets of points from a box centre, in the box's own (u, v) axes."""
    yaw = math.radians(yaw_deg)
    d = np.atleast_2d(np.asarray(xy, float))[:, :2] - np.asarray(centre[:2], float)
    return d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw), -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)


def longest_run(flags) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best
