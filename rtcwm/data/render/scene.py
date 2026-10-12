"""Background traffic and the scene description saved for the coarse proxy."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import carla

# level bounding boxes exported for the coarse proxy: CARLA label -> CWM class
STATIC_LABELS = {"Buildings": 0, "Walls": 0, "Fences": 0, "Poles": 0, "TrafficLight": 0, "TrafficSigns": 0,
                 "Static": 0, "Dynamic": 0, "Bridge": 0, "GuardRail": 0, "Vegetation": 5, "Water": 2}
SCENE_RADIUS_M = 300.0


@dataclass
class Traffic:
    actors: list = field(default_factory=list)
    controllers: list = field(default_factory=list)
    vehicles: list = field(default_factory=list)
    walkers: list = field(default_factory=list)


def spawn_traffic(world, bp, rng, n_vehicles: int, n_walkers: int) -> Traffic:
    """Parked vehicles (physics off: no traffic manager, it segfaults libcarla on some nodes, and a
    static street is what a revisit dataset wants) and AI-driven background pedestrians."""
    t = Traffic()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    vbps = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    for sp in spawns[:n_vehicles]:
        v = world.try_spawn_actor(rng.choice(vbps), sp)
        if v:
            v.set_simulate_physics(False)
            t.actors.append(v)
            t.vehicles.append(v)
    wbps = bp.filter("walker.pedestrian.*")
    ctrl_bp = bp.find("controller.ai.walker")
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(rng.choice(wbps), carla.Transform(loc + carla.Location(z=1.0)))
        if w:
            t.actors.append(w)
            t.walkers.append(w)
            t.controllers.append(world.spawn_actor(ctrl_bp, carla.Transform(), w))
    world.tick()
    for c in t.controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.0 + rng.random() * 0.6)
    return t


def adult_walkers(bp):
    return [b for b in bp.filter("walker.pedestrian.*")
            if not (b.has_attribute("age") and b.get_attribute("age").as_str() == "child")]


def frame_record(hero, walkers) -> dict:
    """Per-frame scene state for the coarse proxy: protagonist bones, background walker poses."""
    bones = {bt.name: [round(bt.world.location.x, 3), round(bt.world.location.y, 3), round(bt.world.location.z, 3)]
             for bt in hero.get_bones().bone_transforms}
    others = []
    for w in walkers:
        wt = w.get_transform()
        others.append([round(wt.location.x, 2), round(wt.location.y, 2), round(wt.location.z, 2), round(wt.rotation.yaw, 1)])
    return dict(bones=bones, walkers=others)


def scene_static(world, centre, hidden, vehicles) -> dict:
    """Level boxes, driving-lane waypoints and parked vehicles within SCENE_RADIUS_M of `centre`.
    Objects switched off for the walk (`hidden`: their box centres) are left out."""
    L = carla.CityObjectLabel
    cx, cy = centre
    boxes = []
    for name, cls in STATIC_LABELS.items():
        if not hasattr(L, name):
            continue
        for bb in world.get_level_bbs(getattr(L, name)):
            if math.hypot(bb.location.x - cx, bb.location.y - cy) > SCENE_RADIUS_M:
                continue
            if any(abs(bb.location.x - h[0]) < 0.05 and abs(bb.location.y - h[1]) < 0.05
                   and abs(bb.location.z - h[2]) < 0.05 for h in hidden):
                continue
            boxes.append(dict(cls=cls, label=name, c=[bb.location.x, bb.location.y, bb.location.z],
                              e=[bb.extent.x, bb.extent.y, bb.extent.z],
                              r=[bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll]))
    lanes = []
    for w in world.get_map().generate_waypoints(2.0):
        p = w.transform.location
        if math.hypot(p.x - cx, p.y - cy) <= SCENE_RADIUS_M:
            lanes.append([p.x, p.y, p.z, w.transform.rotation.yaw, w.lane_width])
    vboxes = []
    for v in vehicles:
        bb, vt = v.bounding_box, v.get_transform()
        c = vt.transform(bb.location)
        vboxes.append(dict(c=[c.x, c.y, c.z], e=[bb.extent.x, bb.extent.y, bb.extent.z],
                           r=[vt.rotation.pitch, vt.rotation.yaw, vt.rotation.roll]))
    return dict(boxes=boxes, lanes=lanes, vehicles=vboxes)
