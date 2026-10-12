"""The protagonist's out -> back -> out walk.

The walk follows the paved middle of one sidewalk lane: out to a turn point b, back past the start,
and out again to b2 beyond b (so the last leg never stalls at its end). Targets every 2 m along that
line are fed to CARLA's walker AI, which routes round street furniture. Posts, signs and tree trunks
standing on the line are switched off for the clip.

History of the rules (dataset_v1 failures they fix):
  * walker AI given only far targets took the grass verge, where lamp posts stand, and walked
    through them (~140 windows)                                   -> dense targets on the paving
  * CARLA's sidewalk lane often spans the verge (centre line on the grass) and at a block's end
    next() runs on across the road                                -> paved_line()
  * Town04 navmesh lies under the terrain (walker rendered underground) -> on_ground(); Town04 is
    dropped from the dataset anyway
  * the AI's own arrival radius is larger than ours: on "arriving" it wanders to a random place
                                                                  -> Steerer re-issues targets
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import carla
import numpy as np

TARGET_STEP_M = 2.0
SWITCH_RADIUS_M = 4.0
REISSUE_EVERY = 48                     # frames
HIDE_RADIUS_M = 0.9
MAX_ATTEMPTS = 300
NAVMESH_FALLBACK_AFTER = 100           # attempts; maps with almost no sidewalk lanes (Town02/04)


@dataclass
class Route:
    kind: str                          # "sidewalk" (paved-line route) or "navmesh" (fallback)
    start: carla.Location
    b: carla.Location
    b2: carla.Location
    leg: float
    targets: list = field(default_factory=list)
    line: list = field(default_factory=list)
    hidden_ids: list = field(default_factory=list)
    hidden_boxes: list = field(default_factory=list)    # box centres of switched-off objects


class RoutePlanner:
    def __init__(self, world, rng):
        self.world, self.rng, self.map = world, rng, world.get_map()
        L = carla.CityObjectLabel
        self.L = L
        self.see_through = {getattr(L, n) for n in ("Pedestrians", "Rider", "Vegetation", "Poles", "TrafficSigns",
                                                    "TrafficLight", "Other") if hasattr(L, n)}
        self.all_nav = [p for p in (world.get_random_location_from_navigation() for _ in range(4000))
                        if p is not None and self.on_ground(p)]
        pool = [p for p in self.all_nav if self.on_sidewalk(p)]
        # start and turn points must lie on a sidewalk: the navmesh also covers lawns and planters
        self.pool = pool if len(pool) >= 50 else self.all_nav

    # --- probes
    def on_sidewalk(self, p) -> bool:
        w = self.map.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
        return w is not None and w.transform.location.distance(p) < 1.5

    def on_ground(self, p) -> bool:
        g = self.world.ground_projection(p + carla.Location(z=3.0), 6.0)
        return g is not None and abs(g.location.z - p.z) < 0.3

    def ground_label(self, q):
        top = q + carla.Location(z=1.5)
        hits = [h for h in self.world.cast_ray(top, q + carla.Location(z=-1.5)) if h.label not in self.see_through]
        return min(hits, key=lambda h: h.location.distance(top)).label if hits else None

    def snap(self, q, tol: float = 1.0):
        c = min(self.all_nav, key=lambda p: (p.x - q.x) ** 2 + (p.y - q.y) ** 2)
        return c if math.hypot(c.x - q.x, c.y - q.y) < tol else None

    def paved_line(self, w, length: float, step: float = TARGET_STEP_M):
        """Points every `step` m along a sidewalk lane, each moved across the lane to the middle of
        its paved part. None if a point has no paving under it or the lane jumps (a junction)."""
        pts, run_m, prev = [], 0.0, None
        while True:
            loc, right = w.transform.location, w.transform.get_right_vector()
            if prev is not None and loc.distance(prev) > 1.5 * step:
                return None
            half = 0.5 * w.lane_width
            offs = np.arange(-half + 0.2, half - 0.2 + 1e-6, 0.3)
            paved = [self.ground_label(loc + right * float(o)) == self.L.Sidewalks for o in offs]
            if not any(paved):
                return None
            best, cur, bs, cs = 0, 0, 0, 0                      # longest run of paved samples
            for i, p in enumerate(paved):
                cur, cs = (cur + 1, cs) if p else (0, i + 1)
                if cur > best:
                    best, bs = cur, cs
            pts.append(loc + right * float(np.mean(offs[bs:bs + best])))
            prev = loc
            if run_m >= length:
                return pts
            nx = w.next(step)
            if not nx:
                return None
            w, run_m = nx[0], run_m + step

    # --- planning
    def _spawn(self, bps, start, b):
        hb = self.rng.choice(bps)
        if hb.has_attribute("is_invincible"):
            hb.set_attribute("is_invincible", "true")
        yaw = math.degrees(math.atan2(b.y - start.y, b.x - start.x))
        return self.world.try_spawn_actor(hb, carla.Transform(start + carla.Location(z=1.0), carla.Rotation(yaw=yaw)))

    def plan(self, leg: float, walker_bps):
        """Spawn the protagonist on a valid route. Returns (hero, Route)."""
        for attempt in range(MAX_ATTEMPTS):
            start = self.rng.choice(self.pool)
            w0 = self.map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
            nb = w0.next(leg) if w0 else []
            nb2 = w0.next(1.4 * leg) if w0 else []
            if (not nb or not nb2) and attempt >= NAVMESH_FALLBACK_AFTER:
                far = [q for q in self.all_nav if 0.75 * leg <= math.hypot(q.x - start.x, q.y - start.y) <= leg]
                if not far:
                    continue
                b = self.rng.choice(far)
                ext = carla.Location(x=start.x + 1.4 * (b.x - start.x), y=start.y + 1.4 * (b.y - start.y), z=b.z)
                b2 = min(self.all_nav, key=lambda q: (q.x - ext.x) ** 2 + (q.y - ext.y) ** 2)
                hero = self._spawn(walker_bps, start, b)
                if hero:
                    return hero, Route("navmesh", start, b, b2, leg, targets=[b, start, b2])
                continue
            if not nb or not nb2:
                continue
            line = self.paved_line(w0, 1.4 * leg)       # the whole walk stays on this lane's paving
            if line is None:
                continue
            n_out = max(1, int(leg / TARGET_STEP_M))
            start, b, b2 = self.snap(line[0]), self.snap(line[n_out]), self.snap(line[-1])
            if start is None or b is None or b2 is None:
                continue
            if math.hypot(b.x - start.x, b.y - start.y) < 0.6 * leg:     # lane bends back on itself
                continue
            hero = self._spawn(walker_bps, start, b)
            if hero:
                route = Route("sidewalk", start, b, b2, leg, targets=[b, start, b2], line=line)
                out_pts = [p for p in (self.snap(q) for q in line[1:n_out + 1]) if p is not None]
                out2_pts = [p for p in (self.snap(q) for q in line[1:]) if p is not None]
                if len(out_pts) >= 3 and len(out2_pts) > len(out_pts):
                    route.targets = out_pts + list(reversed(out_pts[:-1])) + [start] + out2_pts
                self.hide_obstacles(route)
                return hero, route
        raise RuntimeError("could not spawn a protagonist with an out-and-back route")

    def hide_obstacles(self, route: Route):
        """Switch off posts, signs, traffic lights and tree trunks standing on the walking line."""
        xy = np.array([[p.x, p.y] for p in route.line])
        for lab in ("Poles", "TrafficSigns", "TrafficLight", "Vegetation"):
            if not hasattr(self.L, lab):
                continue
            for ob in self.world.get_environment_objects(getattr(self.L, lab)):
                bb = ob.bounding_box
                if lab == "Vegetation" and bb.extent.z < 1.2:
                    continue                                    # hedges / bushes: not on the paving
                if float(np.min(np.hypot(xy[:, 0] - bb.location.x, xy[:, 1] - bb.location.y))) < HIDE_RADIUS_M:
                    route.hidden_ids.append(ob.id)
                    route.hidden_boxes.append([bb.location.x, bb.location.y, bb.location.z])
        if route.hidden_ids:
            self.world.enable_environment_objects(route.hidden_ids, False)


class Steerer:
    """Feeds the route targets to the walker AI: the next target from SWITCH_RADIUS_M out, and the
    current one re-issued every REISSUE_EVERY frames."""

    def __init__(self, controller, hero, targets):
        self.ctrl, self.hero, self.targets, self.i = controller, hero, targets, 0
        self.ctrl.go_to_location(targets[0])

    def step(self, frame: int):
        p = self.hero.get_location()
        t = self.targets[self.i]
        if self.i < len(self.targets) - 1 and math.hypot(p.x - t.x, p.y - t.y) < SWITCH_RADIUS_M:
            self.i += 1
            self.ctrl.go_to_location(self.targets[self.i])
        elif frame % REISSUE_EVERY == 0:
            self.ctrl.go_to_location(t)
