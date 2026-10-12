"""Renderer constants."""
from __future__ import annotations

import numpy as np

from ..geometry import HFOV_DEG

FPS = 24
RGB_W, RGB_H = 1344, 768
SENSOR_W, SENSOR_H = 672, 384          # depth / segmentation, pooled 2x2 to the 336x192 proxy
HFOV = HFOV_DEG                        # CWM proxy intrinsics: f = 206.6 px at 336x192

# CARLA 0.9.15 semantic tags -> CWM's released vocabulary {0,1,2,4,5,8,9,10}. Everything else
# (sidewalk, building, terrain, poles, other pedestrians, ...) stays 0; the protagonist is set to 8
# from the instance channel.
TAG2CWM = np.zeros(256, np.uint8)
TAG2CWM[[1, 24]] = 4                   # road, road line
TAG2CWM[9] = 5                         # vegetation
TAG2CWM[11] = 1                        # sky
TAG2CWM[23] = 2                        # water
TAG2CWM[[14, 15, 16, 17, 18, 19]] = 10  # car, truck, bus, train, motorcycle, bicycle
TAG_PEDESTRIAN = 12
TAGS_SOLID = (3, 4, 5, 9, 20)          # building, wall, fence, vegetation, static -- for the leg check

WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset", "CloudySunset",
            "WetCloudyNoon", "SoftRainNoon"]   # no night presets: CARLA nights render nearly black

WALK_SPEED = 1.4                       # m/s
LEG_PER_SECOND = 0.36 * WALK_SPEED     # out-leg length per clip second (out, back, out again)
WARMUP_S = 1.0
