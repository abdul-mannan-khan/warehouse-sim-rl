"""Replay a simulated shift inside NVIDIA's warehouse digital twin (Isaac Sim 6.2, headless, RTX).

    /isaac-sim/python.sh isaac/replay_isaac.py --trace media/trace.json --out frames/
    /isaac-sim/python.sh isaac/replay_isaac.py --trace media/trace.json --out preview/ --preview

The SimPy event trace drives everything. For each frame we compute where every order is in its life
cycle (waiting, robot driving out, queueing at the picker, being picked, driving back) and place NVIDIA's
iw.hub robots along aisle routes in the `full_warehouse` environment. The four picking zones of the model are
mapped to four rack aisles and the depot to a packing area on the open floor. Pickers and packers are
NVIDIA's characters. While the simulator says a pick is in progress, the picker turns to the rack, takes a
carton off the shelf and places it in the robot's tote (pose.py: spine bend, squat and arm IK on top of a
retargeted idle clip); packers move items from the incoming bin into a shipping box while totes wait to be packed.

The video is cut into five shots (establishing crane, a camera following one robot, a picker's aisle, the
packing area, and an overview of the rest of the shift). Each shot covers a different stretch of the shift at
its own time-lapse rate; overlay.py draws the clock, the rate, KPIs and labels from hud.json.
"""
import argparse
import json
import math
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--out", default="frames/")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--subframes", type=int, default=12)
ap.add_argument("--preview", action="store_true", help="render one frame from the middle of each shot")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--hud-only", action="store_true", help="write hud.json for the frames without rendering")
ap.add_argument("--frames", default="", help="comma-separated frame numbers to render (for checks)")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

from isaacsim import SimulationApp                      # noqa: E402
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": W, "height": H})

import carb                                            # noqa: E402
import numpy as np                                     # noqa: E402
import omni.client                                     # noqa: E402
import omni.replicator.core as rep                     # noqa: E402
import omni.usd                                        # noqa: E402
from PIL import Image                                  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel, Vt  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

ROOT = get_assets_root_path()
ENV = f"{ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
A_ROBOT = f"{ROOT}/Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd"
A_TOTE = f"{ROOT}/Isaac/Props/KLT_Bin/small_KLT_visual.usd"
A_TABLE = f"{ROOT}/Isaac/Props/PackingTable/packing_table.usd"
A_FORKLIFT = f"{ROOT}/Isaac/Props/Forklift/forklift.usd"
A_PALLET = f"{ROOT}/Isaac/Props/Pallet/pallet.usd"
A_BOX = f"{ROOT}/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"
PEOPLE = [f"{ROOT}/Isaac/People/Characters/{c}/{c}.usd" for c in
          ("male_adult_construction_05_new", "male_adult_construction_01_new", "male_adult_construction_03",
           "female_adult_police_03_new")]
ANIM = {"idle": (f"{ROOT}/Isaac/People/Animations/stand_idle_loop.skelanim.usd", None),
        "walk": (f"{ROOT}/Isaac/People/Animations/stand_walk_loop_in_place.skelanim.usd", None)}

T = json.load(open(args.trace))
ORDERS = sorted(T["orders"], key=lambda o: o["created"])
NZ = len(T["zone_distance_m"])
SHIFT = T["minutes"]

# ------------------------------------------------------------------ layout (metres, warehouse frame)
# Racks run along +y between y = 8.8 and 24.9; aisle centre lines sit between rack rows.
AISLE_X = [-22.85, -17.9, -12.95, -8.0, -3.0, 1.95]
ZONE_AISLE = [AISLE_X[3], AISLE_X[2], AISLE_X[1], AISLE_X[0]]      # the four clear aisles; zone 1 nearest
ZONE_PICK_Y = [12.5, 15.5, 18.5, 21.5]
LANE = 0.85                      # out lane on the +x side of an aisle, return lane on the -x side
RACK_FACE = 1.925                 # pick face of the rack on the +x side of each aisle (from the scene bounds)
SHELF_Z = 1.30                    # first shelf level above the floor pallets
PICKER_DX = 1.50                  # picker stands between the robot stop and the rack face
CROSS_OUT, CROSS_BACK = 5.4, 6.8  # two-lane cross aisle in front of the racks
DOCK_Y = -3.5
DOCK_X = [-14.5 + 1.9 * r for r in range(T["n_robots"])]
QUEUE_GAP = 1.6
YAW_OFF = float(os.environ.get("YAW_OFF", -90.0))       # robot asset forward axis vs +x


def path_out(o):
    r, z = o["robot"], o["zone"]
    x_in = ZONE_AISLE[z] + LANE
    return [(DOCK_X[r], DOCK_Y), (DOCK_X[r], CROSS_OUT), (x_in, CROSS_OUT), (x_in, ZONE_PICK_Y[z])]


def path_back(o):
    r, z = o["robot"], o["zone"]
    x_in, x_back = ZONE_AISLE[z] + LANE, ZONE_AISLE[z] - LANE
    py = ZONE_PICK_Y[z]
    return [(x_in, py), (x_in - 0.2, py + 0.9), (x_back, py + 0.9), (x_back, CROSS_BACK),
            (DOCK_X[r] + 0.6, CROSS_BACK), (DOCK_X[r] + 0.6, DOCK_Y + 0.8), (DOCK_X[r], DOCK_Y)]


class Poly:
    def __init__(self, pts):
        self.p = np.array(pts, float)
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(self.cum[-1])

    def at(self, s):
        s = min(max(s, 0.0), self.L)
        i = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.p) - 2)
        f = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        return self.p[i] + (self.p[i + 1] - self.p[i]) * f

    def pose(self, s):
        a, b = self.at(s - 0.45), self.at(s + 0.45)
        if np.linalg.norm(b - a) < 1e-6:
            a, b = self.at(self.L - 0.9), self.at(self.L)
        return self.at(s), math.atan2(b[1] - a[1], b[0] - a[0])


def ease(f):
    f = min(max(f, 0.0), 1.0)
    return f - math.sin(2 * math.pi * f) / (2 * math.pi)


# per-order route geometry and the orders ahead of it in the same zone queue
for o in ORDERS:
    if o["dispatched"] is None:
        continue
    o["P_out"], o["P_back"] = Poly(path_out(o)), Poly(path_back(o))
for o in ORDERS:
    if o.get("at_zone") is None:
        continue
    o["ahead"] = [q for q in ORDERS if q is not o and q["zone"] == o["zone"] and q.get("at_zone") is not None
                  and q["at_zone"] < o["at_zone"] and (q["pick_end"] is None or q["pick_end"] > o["at_zone"])]
    o["slot0"] = len(o["ahead"])


def slot(o, t):
    """Queue position (0 = at the picker), easing forward as robots ahead leave."""
    s = 0.0
    for q in o["ahead"]:
        s += 1.0 if q["pick_end"] is None or t < q["pick_end"] else max(0.0, 1.0 - (t - q["pick_end"]) / 0.12)
    return s


def state_at(t):
    waiting = [[] for _ in range(NZ)]
    robots = {}
    picking = [None] * NZ
    done = late = 0
    ct_sum = 0.0
    for o in ORDERS:
        if o["created"] > t:
            break
        if o["dispatched"] is None or o["dispatched"] > t:
            waiting[o["zone"]].append(o["rush"])
            continue
        if o["done"] is not None and o["done"] <= t:
            done += 1
            ct = o["done"] - o["created"]
            ct_sum += ct
            late += ct > (T["sla_rush_min"] if o["rush"] else T["sla_normal_min"])
        if o["back"] is not None and t >= o["back"]:
            continue
        z = o["zone"]
        if o["at_zone"] is not None and t < o["at_zone"]:
            span = max(o["at_zone"] - o["dispatched"], 1e-9)
            f = (t - o["dispatched"]) / span
            turn = min(0.12, 0.25 / max(span, 1e-3))          # first moments: turn on the spot at the dock
            P = o["P_out"]
            end = P.L - QUEUE_GAP * o["slot0"]
            if f < turn:
                pos, _ = P.pose(0.0)
                h0, h1 = -math.pi / 2, math.pi / 2
                k = ease(f / turn)
                heading = h0 + (h1 - h0) * k
            else:
                pos, heading = P.pose(end * ease((f - turn) / (1 - turn)))
            phase = "out"
        elif o["pick_end"] is None or t < o["pick_end"]:
            k = slot(o, t)
            pos = np.array([ZONE_AISLE[z] + LANE, ZONE_PICK_Y[z] - QUEUE_GAP * k])
            heading = math.pi / 2
            phase = "pick" if (o["pick_start"] is not None and t >= o["pick_start"] and k < 0.05) else "queue"
            if phase == "pick":
                picking[z] = o
        else:
            back = o["back"] if o["back"] is not None else o["pick_end"] + (o["at_zone"] - o["dispatched"])
            f = (t - o["pick_end"]) / max(back - o["pick_end"], 1e-9)
            pos, heading = o["P_back"].pose(o["P_back"].L * ease(f))
            phase = "back"
        robots[o["robot"]] = (pos, heading, phase, o["rush"], z, o)
    packing = sum(1 for o in ORDERS if o["back"] is not None and o["back"] <= t
                  and (o["done"] is None or o["done"] > t))
    return dict(waiting=waiting, robots=robots, picking=picking, packing=packing, done=done, late=late,
                ct=(ct_sum / done if done else 0.0))


# ------------------------------------------------------------------ shots and the sim-time schedule
FPS = args.fps


def busiest(t_from, span, score, step=0.25, t_to=None):
    best, bt = -1, t_from
    t = t_from
    while t <= (t_to or SHIFT - span - 30):
        s = score(t, t + span)
        if s > best:
            best, bt = s, t
        t += step
    return bt


SHOTS = []                                    # (name, seconds, sim_start, sim_end, extra)
t = 0.0
SHOTS.append(("establish", 9.0, 0.0, 40.0, {}))
t = 40.0
follow = next(o for o in ORDERS if o["dispatched"] is not None and o["dispatched"] >= 45.0 and o["zone"] >= 2
              and o["back"] is not None)
fs = follow["dispatched"] - 0.04
SHOTS.append(("follow", 14.0, fs, fs + 14.0 * 10 / 60, {"robot": follow["robot"], "order": follow}))


def aisle_score(z, a, b):
    """Share of the window with a pick long enough to show the full shelf-to-tote motion, plus arrivals."""
    cover = sum(max(0.0, min(o["pick_end"], b) - max(o["pick_start"], a)) for o in ORDERS
                if o["zone"] == z and o["pick_start"] is not None and o["pick_end"] is not None
                and (o["pick_end"] - o["pick_start"]) * 5.0 >= 4.2)
    arrivals = sum(1 for o in ORDERS if o["zone"] == z and o.get("at_zone") is not None and a < o["at_zone"] < b)
    return cover / (b - a) + 0.15 * min(arrivals, 3)


AISLE_ZONE, a0 = max(((z, busiest(SHOTS[-1][3] + 1, 2.0, lambda a, b, z=z: aisle_score(z, a, b), step=0.1, t_to=200))
                      for z in range(NZ)), key=lambda zt: aisle_score(zt[0], zt[1], zt[1] + 2.0))
SHOTS.append(("aisle", 10.0, a0, a0 + 2.0, {"zone": AISLE_ZONE}))
d0 = busiest(SHOTS[-1][3] + 1, 1.8, lambda a, b: sum(1 for o in ORDERS if o["back"] and a + 0.2 < o["back"] < b)
             + sum(1 for o in ORDERS if o["dispatched"] and a + 0.2 < o["dispatched"] < b), t_to=260)
SHOTS.append(("depot", 9.0, d0, d0 + 1.8, {}))
SHOTS.append(("overview", 18.0, SHOTS[-1][3] + 0.5, SHIFT, {}))
N = int(round(sum(s[1] for s in SHOTS) * FPS))

PLAN = []                                     # per frame: (shot index, sim minute, local fraction)
for i, (name, sec, a, b, _) in enumerate(SHOTS):
    n = int(round(sec * FPS))
    for k in range(n):
        f = k / max(n - 1, 1)
        PLAN.append((i, a + (b - a) * f, f))
PLAN = PLAN[:N]
print("shots:", [(s[0], s[1], round(s[2], 1), round(s[3], 1)) for s in SHOTS], "frames", len(PLAN), flush=True)

# ------------------------------------------------------------------ stage
omni.usd.get_context().open_stage(ENV)
for _ in range(30):
    app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")

settings = carb.settings.get_settings()
for k, v in {"/rtx/indirectDiffuse/enabled": True, "/rtx/ambientOcclusion/enabled": True,
             "/rtx/reflections/enabled": True, "/rtx/directLighting/sampledLighting/enabled": True,
             "/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 2}.items():
    settings.set(k, v)

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def bbox(prim):
    cache.Clear()
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


# cut away the roof so the high shots can see in (walls stay)
roof = 0
for p in stage.GetDefaultPrim().GetChildren():
    n = p.GetName()
    if n.startswith(("SM_CeilingA", "SM_BeamA", "SM_LampCeiling", "SM_PillarPartA", "SM_PillarA")):
        UsdGeom.Imageable(p).MakeInvisible()
        roof += 1
print("roof prims hidden:", roof, flush=True)

_mats = {}


def material(name, rgb, emit=0.0, rough=0.5, metal=0.0):
    if name in _mats:
        return _mats[name]
    path = f"/World/Looks/{name}"
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/s")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if emit:
        s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * emit for c in rgb]))
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    _mats[name] = m
    return m


def bind(prim, m):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.strongerThanDescendants)


def xform(path, pos=(0, 0, 0), yaw=0.0):
    x = UsdGeom.Xform.Define(stage, path)
    t_ = x.AddTranslateOp()
    r_ = x.AddRotateZOp()
    t_.Set(Gf.Vec3d(*map(float, pos)))
    r_.Set(float(yaw))
    return x, t_, r_


def reference(path, url, pos=(0, 0, 0), yaw=0.0):
    x, t_, r_ = xform(path, pos, yaw)
    child = stage.DefinePrim(path + "/asset", "Xform")
    child.GetReferences().AddReference(url)
    return x, t_, r_


def fit(path, size, axis="max", base=True):
    """Scale a referenced asset so its largest side (or height, axis='z') is `size` m; stand it on z of its xform."""
    child = stage.GetPrimAtPath(path + "/asset")
    g = bbox(child)
    s = g.GetSize()
    ref = s[2] if axis == "z" else max(s)
    if ref <= 0:
        print("fit: empty bbox", path, flush=True)
        return 1.0
    k = size / ref
    UsdGeom.Xformable(child).AddScaleOp(opSuffix="fit").Set(Gf.Vec3f(k, k, k))
    if base:
        g2 = bbox(child)
        top = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0]
        t_ = top.Get()
        top.Set(Gf.Vec3d(t_[0], t_[1], t_[2] + (t_[2] - g2.GetMin()[2])))
    return k


M = {
    "normal": material("normal", (0.08, 0.32, 0.95), rough=0.35, emit=0.35),
    "rush": material("rush", (0.95, 0.12, 0.08), rough=0.35, emit=0.9),
    "pick": material("pick", (0.1, 0.85, 0.3), rough=0.35, emit=1.2),
    "idle": material("idle", (0.5, 0.52, 0.55), rough=0.5),
    "lane": material("lane", (0.95, 0.75, 0.05), rough=0.6, emit=0.15),
    "dock": material("dock", (0.1, 0.55, 0.95), rough=0.6, emit=0.2),
}


def strip(path, a, b, w, m):
    a, b = np.array(a, float), np.array(b, float)
    v = b - a
    mid = (a + b) / 2
    x, _, _ = xform(path, (mid[0], mid[1], 0.006), math.degrees(math.atan2(v[1], v[0])))
    c = UsdGeom.Cube.Define(stage, path + "/geo")
    c.CreateSizeAttr(1.0)
    c.AddScaleOp().Set(Gf.Vec3f(float(np.linalg.norm(v)) + w, w, 0.006))
    bind(c.GetPrim(), m)


# AMR travel lanes painted on the floor (cross aisle + zone aisles)
strip("/World/lanes/cross_out", (min(ZONE_AISLE) - 1, CROSS_OUT), (max(DOCK_X) + 1, CROSS_OUT), 0.07, M["lane"])
strip("/World/lanes/cross_back", (min(ZONE_AISLE) - 1, CROSS_BACK), (max(DOCK_X) + 1, CROSS_BACK), 0.07, M["lane"])
for r, x in enumerate(DOCK_X):                                       # charging / dock pads
    strip(f"/World/lanes/dock{r}a", (x - 0.55, DOCK_Y - 0.75), (x + 0.55, DOCK_Y - 0.75), 0.06, M["dock"])
    strip(f"/World/lanes/dock{r}b", (x - 0.55, DOCK_Y + 0.75), (x + 0.55, DOCK_Y + 0.75), 0.06, M["dock"])

# packing area: benches south of the docks with packers, pallets of outbound boxes, a parked forklift
PACK_Y = DOCK_Y - 2.8
TABLE_X = [DOCK_X[0] + 0.9, (DOCK_X[0] + DOCK_X[-1]) / 2, DOCK_X[-1] - 0.9]
for i, x in enumerate(TABLE_X):
    reference(f"/World/pack/table{i}", A_TABLE, (x, PACK_Y, 0.0), 0.0)
for i, (x, y) in enumerate([(-23.0, -5.0), (-23.0, -7.2), (-21.0, -5.0), (-21.0, -7.5), (-1.5, -9.0), (0.5, -9.0)]):
    reference(f"/World/pal/p{i}", A_PALLET, (x, y, 0.0), 90.0 * (i % 2))
    for j in range(3 if i % 3 else 2):
        reference(f"/World/pal/p{i}b{j}", A_BOX, (x - 0.3 + 0.6 * (j % 2), y + 0.25 - 0.5 * (j // 2), 0.16), 0.0)
reference("/World/forklift", A_FORKLIFT, (-22.5, -11.5, 0.0), 35.0)

# pickers: one per zone, standing between the robot stop and the rack face
CHARS, CHAR_URL, CHAR_OPS = [], [], []
for z in range(NZ):
    path, url = f"/World/picker{z}", PEOPLE[z % len(PEOPLE)]
    CHAR_OPS.append(reference(path, url, (ZONE_AISLE[z] + PICKER_DX, ZONE_PICK_Y[z], 0.0), 0.0))
    CHARS.append(path)
    CHAR_URL.append(url)

M["card"] = material("card", (0.60, 0.43, 0.25), rough=0.85)
M["tape"] = material("tape", (0.80, 0.68, 0.45), rough=0.4)
M["bin"] = material("bin", (0.25, 0.42, 0.62), rough=0.5)
CART = (0.17, 0.21, 0.14)                         # parcel: depth, width, height (m)


def carton(path, size=CART):
    x, t_, r_ = xform(path)
    b = UsdGeom.Cube.Define(stage, path + "/box")
    b.CreateSizeAttr(1.0)
    b.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(b.GetPrim(), M["card"])
    tp = UsdGeom.Cube.Define(stage, path + "/tape")
    tp.CreateSizeAttr(1.0)
    tp.AddTranslateOp().Set(Gf.Vec3d(0, 0, size[2] / 2))
    tp.AddScaleOp().Set(Gf.Vec3f(size[0] + 0.004, 0.05, 0.006))
    bind(tp.GetPrim(), M["tape"])
    return {"prim": x.GetPrim(), "t": t_, "r": r_}


def show(prim, on):
    im = UsdGeom.Imageable(prim)
    im.MakeVisible() if on else im.MakeInvisible()


def open_box(path, c, size=(0.40, 0.34, 0.24), wall=0.012):
    """Open shipping carton on the packing table; parcels are lowered into it."""
    sx, sy, sz = size
    parts = [((0, 0, wall / 2), (sx, sy, wall)),
             ((sx / 2, 0, sz / 2), (wall, sy, sz)), ((-sx / 2, 0, sz / 2), (wall, sy, sz)),
             ((0, sy / 2, sz / 2), (sx, wall, sz)), ((0, -sy / 2, sz / 2), (sx, wall, sz)),
             ((sx / 2 + 0.09, 0, sz + 0.04), (0.18, sy, wall))]                  # one flap folded out
    xform(path, c)
    for k, (o, sc) in enumerate(parts):
        q = UsdGeom.Cube.Define(stage, f"{path}/w{k}")
        q.CreateSizeAttr(1.0)
        q.AddTranslateOp().Set(Gf.Vec3d(*o))
        if k == 5:
            q.AddRotateYOp().Set(-35.0)
        q.AddScaleOp().Set(Gf.Vec3f(*sc))
        bind(q.GetPrim(), M["card"])


# one carton per worker (on the shelf / in the hands); three parcel slots in every robot's tote
HAND = [carton(f"/World/parcels/w{k}") for k in range(NZ + 3)]

robots = []
for r in range(T["n_robots"]):
    x, tr, rot = reference(f"/World/robot{r}", A_ROBOT, (DOCK_X[r], DOCK_Y, 0.0), -90.0 + YAW_OFF)
    robots.append({"x": x, "t": tr, "r": rot, "items": [carton(f"/World/robot{r}/item{k}") for k in range(3)]})

# lights: keep the warehouse's own ceiling lights, add sky and a soft sun through the open roof
dome = UsdLux.DomeLight.Define(stage, "/World/sky")
dome.CreateIntensityAttr(float(os.environ.get("DOME", 400)))
dome.CreateColorAttr(Gf.Vec3f(0.86, 0.91, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(float(os.environ.get("SUN", 1500)))
sun.CreateAngleAttr(2.0)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40.0, 15.0, -35.0))

for _ in range(120):                          # stream referenced assets before measuring them
    app.update()

for i in range(3):
    fit(f"/World/pack/table{i}", 2.0)
    g = bbox(stage.GetPrimAtPath(f"/World/pack/table{i}/asset")).GetSize()
    if g[1] > g[0]:                                         # long side along x, facing the docks
        UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/pack/table{i}")).GetOrderedXformOps()[1].Set(90.0)
for p in stage.GetPrimAtPath("/World/pal").GetChildren():
    fit(str(p.GetPath()), 1.2 if "b" not in p.GetName() else 0.5)
fit("/World/forklift", 3.2)

# packing stations: an incoming bin and an open shipping box on each table, the packer in front of it
for _ in range(10):
    app.update()
tb = bbox(stage.GetPrimAtPath("/World/pack/table0/asset"))
TABLE_TOP, TABLE_FRONT = float(tb.GetMax()[2]), float(tb.GetMin()[1]) - PACK_Y
print("table", tb.GetSize(), "top", TABLE_TOP, "front", TABLE_FRONT, flush=True)
STATIONS = []
for i, x in enumerate(TABLE_X):
    fy = PACK_Y + TABLE_FRONT
    sup = np.array([x - 0.55, fy + 0.30, TABLE_TOP + 0.09])
    box = np.array([x + 0.20, fy + 0.30, TABLE_TOP + 0.10])
    reference(f"/World/pack/bin{i}", A_TOTE, (sup[0], sup[1], TABLE_TOP + 0.005), 0.0)
    open_box(f"/World/pack/ship{i}", (box[0], box[1], TABLE_TOP + 0.003))
    path, url = f"/World/pack/packer{i}", PEOPLE[(NZ + i) % len(PEOPLE)]
    CHAR_OPS.append(reference(path, url, (x - 0.15, fy - 0.36, 0.0), 0.0))
    CHARS.append(path)
    CHAR_URL.append(url)
    STATIONS.append((sup, box))
for _ in range(60):
    app.update()
for i in range(3):
    fit(f"/World/pack/bin{i}", 0.42)
    bind(stage.GetPrimAtPath(f"/World/pack/bin{i}"), M["bin"])
for c in CHARS:
    fit(c, 1.76, axis="z")
rb = bbox(stage.GetPrimAtPath("/World/robot0/asset"))
print("robot bbox", rb.GetSize(), rb.GetMin(), rb.GetMax(), flush=True)
# the asset origin is at the wheel axles, not the floor: lift the robots so the wheels sit on the floor
ROBOT_Z = max(0.0, -float(rb.GetMin()[2])) if rb.GetSize()[0] > 0 else 0.0
ROBOT_TOP = float(rb.GetMax()[2]) if rb.GetSize()[0] > 0 else 0.35     # in the robot frame
for r, rbt in enumerate(robots):
    rbt["t"].Set(Gf.Vec3d(DOCK_X[r], DOCK_Y, ROBOT_Z))
print("robot lift", ROBOT_Z, flush=True)
for r, rbt in enumerate(robots):
    path = f"/World/robot{r}/tote"
    reference(path, A_TOTE, (0.0, 0.0, ROBOT_TOP + 0.01), 90.0 - YAW_OFF)
    fit(path, 0.6)
    tote_geo = stage.GetPrimAtPath(path)
    bind(tote_geo, M["idle"])
    rbt["tote"] = tote_geo
tg = bbox(robots[0]["tote"])
print("tote bbox", tg.GetMin(), tg.GetMax(), flush=True)
ITEM_Z = ROBOT_TOP + 0.045 + CART[2] / 2
SLOTS = [(-0.09, 0.0, 0.0), (0.09, 0.0, 0.0), (0.0, 0.0, CART[2])]
for rbt in robots:
    for k, it in enumerate(rbt["items"]):
        it["t"].Set(Gf.Vec3d(SLOTS[k][0], SLOTS[k][1], ITEM_Z + SLOTS[k][2]))
        it["r"].Set(0.0)
        show(it["prim"], False)


def robot_point(pos, heading, off):
    """A point given in a robot's local frame (as its tote slots are), in world coordinates."""
    a = heading + math.radians(YAW_OFF)
    c, s_ = math.cos(a), math.sin(a)
    return np.array([pos[0] + c * off[0] - s_ * off[1], pos[1] + s_ * off[0] + c * off[1], off[2] + ROBOT_Z])


# ------------------------------------------------------------------ character animation
# NVIDIA's clips target the biped rig; the characters use a Reallusion rig, so retarget them (retarget.py).
# There is no pick or place clip, so work motions are posed procedurally on the idle clip (pose.py).
import sys                                             # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import Retargeter                        # noqa: E402
from pose import Poser, smooth as sm                   # noqa: E402

BIPED = f"{ROOT}/Isaac/People/Characters/Biped_Setup.usd"
RT, CLIPS, WORKERS = {}, {}, []
xcache = UsdGeom.XformCache()
for i, c in enumerate(CHARS):
    skel = next(p for p in Usd.PrimRange(stage.GetPrimAtPath(c)) if p.IsA(UsdSkel.Skeleton))
    url = CHAR_URL[i]
    if url not in RT:
        RT[url] = Retargeter(BIPED, skel)
        for k, (clip_url, max_s) in ANIM.items():
            CLIPS[(url, k)] = RT[url].clip_mats(clip_url, FPS, max_s)
        print("retargeted", url.rsplit("/", 1)[-1], len(RT[url].tmap), "joints", flush=True)
    r = RT[url]
    a = UsdSkel.Animation.Define(stage, f"/World/anim/a{i}")
    a.GetJointsAttr().Set(r.tjoints)
    a.GetTranslationsAttr().Set(r.t_trans)
    a.GetScalesAttr().Set(r.t_scale)
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([a.GetPath()])
    _, t_op, r_op = CHAR_OPS[i]
    xcache.Clear()
    skel_to_root = xcache.GetLocalToWorldTransform(skel) * xcache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(c)).GetInverse()
    ps = Poser(r, skel_to_root)
    root = t_op.Get()
    WORKERS.append({"a": a, "url": url, "poser": ps, "t": t_op, "r": r_op, "xy": np.array([root[0], root[1]]),
                    "z0": float(root[2]), "fwd": math.atan2(ps.fwd_r[1], ps.fwd_r[0]), "phase": int(FPS * 1.37 * i),
                    "carton": HAND[i]})
    print("worker", c, "body forward in root frame", np.round(ps.fwd_r, 3), "arm", round(ps.arm_len, 3),
          "rest foot h", round(ps.foot_h0, 3), flush=True)
    ps.pose(CLIPS[(url, "idle")][0])
    print("  idle clip foot correction dz", round(ps.last_dz, 3), flush=True)

# a work cycle: turn to the pick point A, reach, grip, lift, turn to the place point B, lower, release, step back
PHASES = [("turnA", 0.00, 0.15), ("reach", 0.15, 0.29), ("grip", 0.29, 0.34), ("lift", 0.34, 0.44),
          ("turnB", 0.44, 0.60), ("lower", 0.60, 0.73), ("release", 0.73, 0.78), ("ret", 0.78, 0.89),
          ("rest", 0.89, 1.0)]
CARRY = np.array([0.30, 0.0, 1.02])                # parcel held in front of the chest (body frame)


def wrists(c):
    """Wrist targets that hold a parcel centred at body-frame point c."""
    hw = CART[1] / 2 + 0.035
    return {"L": (c[0] - 0.05, c[1] + hw, c[2]), "R": (c[0] - 0.05, c[1] - hw, c[2])}


def to_body(W, p, psi):
    d = np.asarray(p[:2]) - W["xy"]
    return np.array([d[0] * math.cos(psi) + d[1] * math.sin(psi),
                     -d[0] * math.sin(psi) + d[1] * math.cos(psi), p[2] - W["z0"]])


def to_world(W, b, psi):
    return np.array([W["xy"][0] + b[0] * math.cos(psi) - b[1] * math.sin(psi),
                     W["xy"][1] + b[0] * math.sin(psi) + b[1] * math.cos(psi), b[2] + W["z0"]])


def heading_to(W, p):
    d = np.asarray(p[:2]) - W["xy"]
    return math.atan2(d[1], d[0])


def work(W, u, A, B, turn_sign=0):
    """Pose for fraction u of a work cycle moving a parcel from A to B (world points).
    Returns facing, posture, hand target, leg blend, where the parcel is and whether it has been placed."""
    psiB = heading_to(W, B)
    dA = heading_to(W, A) - psiB
    dA = (dA + math.pi) % (2 * math.pi) - math.pi
    if turn_sign and dA * turn_sign < 0:
        dA += 2 * math.pi * turn_sign
    psiA = psiB + dA
    bA, bB = to_body(W, A, psiA), to_body(W, B, psiB)
    pA, pB = W["poser"].solve(wrists(bA)), W["poser"].solve(wrists(bB))
    name, k = "rest", 0.0
    for n, a0, a1 in PHASES:
        if a0 <= u < a1:
            name, k = n, (u - a0) / (a1 - a0)
            break
    s = sm(k)
    z = (0.0, 0.0)

    def lerp2(p, q, f_):
        return (p[0] + (q[0] - p[0]) * f_, p[1] + (q[1] - p[1]) * f_)

    leg = 1.0 if abs(dA) > 1.0 else 0.5
    out = {"psi": psiB, "post": z, "tgt": None, "w": 0.0, "legs": 0.0, "parcel": "A", "placed": False}
    if name == "turnA":
        out.update(psi=psiB + dA * s, legs=math.sin(math.pi * k) * leg)
    elif name == "reach":
        out.update(psi=psiA, post=lerp2(z, pA, s), tgt=bA, w=s)
    elif name == "grip":
        out.update(psi=psiA, post=pA, tgt=bA, w=1.0, parcel="A" if k < 0.5 else "hands")
    elif name == "lift":
        out.update(psi=psiA, post=lerp2(pA, z, s), tgt=bA + (CARRY - bA) * s, w=1.0, parcel="hands")
    elif name == "turnB":
        out.update(psi=psiA - dA * s, tgt=CARRY, w=1.0, parcel="hands", legs=math.sin(math.pi * k) * leg)
    elif name == "lower":
        out.update(post=lerp2(z, pB, s), tgt=CARRY + (bB - CARRY) * s, w=1.0, parcel="hands")
    elif name == "release":
        out.update(post=pB, tgt=bB, w=1.0, parcel="hands" if k < 0.5 else None, placed=k >= 0.5)
    elif name == "ret":
        out.update(post=lerp2(pB, z, s), tgt=bB, w=1.0 - s, parcel=None, placed=True)
    else:
        out.update(parcel=None, placed=True)
    return out


def drive(W, sec, m=None, idle_psi=0.0):
    """Apply a work pose (or idle, when m is None) to a worker's skeleton, root and parcel."""
    frames = CLIPS[(W["url"], "idle")]
    base = frames[(int(sec * FPS) + W["phase"]) % len(frames)]
    walk = CLIPS[(W["url"], "walk")]
    ps = W["poser"]
    if m is None:
        L, Tr = ps.pose(base)
        psi = idle_psi
    else:
        psi = m["psi"]
        hands = None
        if m["tgt"] is not None:
            hands = {sd: (np.array(p), m["w"]) for sd, p in wrists(m["tgt"]).items()}
        L, Tr = ps.pose(base, m["post"][0], hands, walk[int(sec * FPS * 1.3) % len(walk)], m["legs"],
                        squat=m["post"][1])
    W["a"].GetRotationsAttr().Set(Poser.quats(L))
    W["a"].GetTranslationsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, t)) for t in Tr]))
    W["r"].Set(math.degrees(psi - W["fwd"]))
    W["t"].Set(Gf.Vec3d(float(W["xy"][0]), float(W["xy"][1]), W["z0"] + ps.last_dz))   # feet on the floor
    pc = W["carton"]
    where = None if m is None else m["parcel"]
    p, yaw = None, psi
    if where == "A":
        p, yaw = m["A"], m.get("A_yaw", psi)
    elif where == "hands":
        p = to_world(W, ps.hand_mid(L, Tr), psi)
    show(pc["prim"], p is not None)
    if p is not None:
        pc["t"].Set(Gf.Vec3d(*map(float, p)))
        pc["r"].Set(math.degrees(yaw))


def pick_cycles(o, rate):
    """Shelf-to-tote cycles shown for an order's pick at this time-lapse rate (0: too fast to show)."""
    end = o["pick_end"] if o["pick_end"] is not None else SHIFT          # still being picked at the end of the shift
    pv = (end - o["pick_start"]) * 60.0 / rate
    if pv < 2.4:
        return 0
    return max(1, int(pv / 4.2))


def shelf_point(z, k):
    dy = (-0.32, 0.18, 0.38, -0.08)[k % 4]
    return np.array([ZONE_AISLE[z] + RACK_FACE + 0.03, ZONE_PICK_Y[z] + dy, SHELF_Z + CART[2] / 2 + 0.005])


PACK_T = 4.6                                       # seconds per packing cycle
SHOT_START = {}
for _f, _p in enumerate(PLAN):
    SHOT_START.setdefault(_p[0], _f)


def animate_workers(f, st, rate):
    """Pose pickers and packers for frame f; returns the number of parcels in each robot's tote."""
    sec = f / FPS
    items = {}
    for r, (pos, heading, phase, rush, z, o) in st["robots"].items():
        if phase == "back":
            items[r] = min(max(pick_cycles(o, rate), 1), 3)
    for z in range(NZ):
        W = WORKERS[z]
        o = st["picking"][z]
        psi_idle = math.pi                                  # facing the aisle and the robot stop
        if o is None:
            drive(W, sec, None, psi_idle)
            continue
        pos, heading = st["robots"][o["robot"]][:2]
        n = pick_cycles(o, rate)
        end = o["pick_end"] if o["pick_end"] is not None else SHIFT
        frac = (PLAN[f][1] - o["pick_start"]) / max(end - o["pick_start"], 1e-9)
        if n == 0:
            items[o["robot"]] = 1 if frac > 0.5 else 0
            drive(W, sec, None, psi_idle)
            continue
        c = min(max(frac, 0.0) * n, n - 1e-6)
        idx, u = int(c), c - int(c)
        slot = SLOTS[min(idx, 2)]
        A = shelf_point(z, idx)
        B = robot_point(pos, heading, (slot[0], slot[1], ITEM_Z + slot[2] + 0.01))
        m = work(W, u, A, B, turn_sign=-1)                 # turn through the up-aisle side
        m["A"], m["A_yaw"] = A, 0.0
        items[o["robot"]] = min(idx + (1 if m["placed"] else 0), 3)
        drive(W, sec, m, psi_idle)
    for i, (sup, box) in enumerate(STATIONS):
        W = WORKERS[NZ + i]
        cyc = (sec + 1.3 * i) / PACK_T
        u = cyc - int(cyc)
        f0 = max(f - int(u * PACK_T * FPS), SHOT_START[PLAN[f][0]])
        if rate < 30 and state_at(PLAN[f0][1])["packing"] > i:
            m = work(W, u, sup, box)
            m["A"], m["A_yaw"] = sup, 0.0
            drive(W, sec, m)
        else:
            drive(W, sec, None, heading_to(W, box))
    return items


# ------------------------------------------------------------------ camera
cam = UsdGeom.Camera.Define(stage, "/World/cam")
HAPER = 20.955
cam.CreateHorizontalApertureAttr(HAPER)
cam.CreateVerticalApertureAttr(HAPER * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
cam_focal = cam.CreateFocalLengthAttr(18.0)
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def lerp(a, b, f):
    return tuple(a[i] + (b[i] - a[i]) * f for i in range(len(a)))


def smooth(f):
    f = min(max(f, 0.0), 1.0)
    return f * f * (3 - 2 * f)


CENTER_X = (min(ZONE_AISLE) + max(DOCK_X)) / 2


def precompute_follow():
    """Chase camera for the follow shot: smoothed behind-and-above the robot."""
    i = next(k for k, s in enumerate(SHOTS) if s[0] == "follow")
    rid = SHOTS[i][4]["robot"]
    out, pos, fwd = {}, None, None
    for f, (si, t_, _) in enumerate(PLAN):
        if si != i:
            continue
        st = state_at(t_)
        if rid in st["robots"]:
            p, h, *_ = st["robots"][rid]
        else:
            p, h = np.array([DOCK_X[rid], DOCK_Y]), -math.pi / 2
        d = np.array([math.cos(h), math.sin(h)])
        pos = p.copy() if pos is None else pos + (p - pos) * 0.25
        fwd = d.copy() if fwd is None else fwd + (d - fwd) * 0.06
        fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
        eye = (pos[0] - fwd[0] * 4.2 + fwd[1] * 1.2, pos[1] - fwd[1] * 4.2 - fwd[0] * 1.2, 2.6)
        tgt = (pos[0] + fwd[0] * 2.5, pos[1] + fwd[1] * 2.5, 0.5)
        out[f] = (eye, tgt, 16.0)
    return out


FOLLOW = precompute_follow()


def camera_for(f):
    si, t_, u = PLAN[f]
    name, ex = SHOTS[si][0], SHOTS[si][4]
    if name == "establish":
        k = smooth(u)
        eye = lerp((-44.0, -58.0, 46.0), (-24.0, -26.0, 17.0), k)
        tgt = lerp((-9.0, 4.0, 0.0), (-8.0, 9.0, 1.5), k)
        return eye, tgt, 18.0
    if name == "follow":
        return FOLLOW[f]
    if name == "aisle":
        z = ex["zone"]
        ax, py = ZONE_AISLE[z], ZONE_PICK_Y[z]
        k = smooth(u)
        eye = lerp((ax - 1.3, py + 3.9, 2.2), (ax - 1.0, py + 3.0, 1.9), k)
        return eye, (ax + 1.0, py - 1.2, 0.85), 18.0
    if name == "depot":
        k = smooth(u)
        mid = (DOCK_X[0] + DOCK_X[-1]) / 2
        eye = lerp((mid + 3.0, PACK_Y - 7.6, 4.3), (mid + 0.8, PACK_Y - 8.0, 4.0), k)
        return eye, (mid - 0.3, PACK_Y + 1.6, 0.4), 18.0
    k = smooth(u)                                                # overview
    eye = lerp((CENTER_X + 9.0, -27.0, 27.0), (CENTER_X - 7.0, -27.0, 27.0), k)
    return eye, (CENTER_X, 8.0, 0.0), 18.0


def set_camera(eye, tgt, focal):
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
    cam_op.Set(view.GetInverse())
    cam_focal.Set(float(focal))
    return view


def project(view, focal, p):
    c = view.Transform(Gf.Vec3d(*p))
    if c[2] > -0.1:
        return None
    k = focal / HAPER * W
    x, y = W / 2 + k * c[0] / -c[2], H / 2 - k * c[1] / -c[2]
    return [x, y] if 0 <= x <= W and 0 <= y <= H else None


rp = rep.create.render_product("/World/cam", (W, H))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])


# ------------------------------------------------------------------ per frame
def apply(f):
    si, t_, u = PLAN[f]
    st = state_at(t_)
    for r, R in enumerate(robots):
        if r in st["robots"]:
            pos, heading, phase, rush, z, _ = st["robots"][r]
            R["t"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), ROBOT_Z))
            R["r"].Set(math.degrees(heading) + YAW_OFF)
            bind(R["tote"], M["pick"] if phase == "pick" else (M["rush"] if rush else M["normal"]))
        else:
            R["t"].Set(Gf.Vec3d(DOCK_X[r], DOCK_Y, ROBOT_Z))
            R["r"].Set(-90.0 + YAW_OFF)
            bind(R["tote"], M["idle"])
    name = SHOTS[si][0]
    rate = (SHOTS[si][3] - SHOTS[si][2]) * 60.0 / SHOTS[si][1]
    items = animate_workers(f, st, rate)
    for r, R in enumerate(robots):
        for k, it in enumerate(R["items"]):
            show(it["prim"], k < items.get(r, 0))
    eye, tgt, focal = camera_for(f)
    view = set_camera(eye, tgt, focal)
    labels = {}
    if name in ("establish", "overview"):
        labels = {f"Zone {z + 1}": project(view, focal, (ZONE_AISLE[z], 9.0, 6.6)) for z in range(NZ)}
        labels["Packing + docks"] = project(view, focal, ((DOCK_X[0] + DOCK_X[-1]) / 2, PACK_Y - 2.2, 0.0))
    elif name == "aisle":
        labels = {f"Zone {ex_z + 1} picker": project(view, focal, (ZONE_AISLE[ex_z] + PICKER_DX, ZONE_PICK_Y[ex_z], 2.1))
                  for ex_z in [SHOTS[si][4]["zone"]]}
    elif name == "follow":
        rid = SHOTS[si][4]["robot"]
        if rid in st["robots"]:
            p = st["robots"][rid][0]
            labels = {f"Robot {rid + 1}": project(view, focal, (p[0], p[1], 0.9))}
    ex = SHOTS[si][4]
    if name == "follow":
        caption = "Following robot {} to Zone {} ({} order)".format(ex["robot"] + 1, ex["order"]["zone"] + 1,
                                                                   "rush" if ex["order"]["rush"] else "normal")
    else:
        caption = {"establish": "NVIDIA warehouse digital twin: 4 picking zones, packing area, 6 iw.hub robots",
                   "aisle": f"Zone {ex.get('zone', 0) + 1}: the picker loads each robot's tote while the next ones queue",
                   "depot": "Packing area: picked items are packed for shipping; idle robots wait at their docks",
                   "overview": "Rest of the shift"}[name]
    return {"frame": f, "shot": name, "u": u, "t": t_, "rate": rate, "caption": caption, "done": st["done"],
            "late": st["late"], "ct": st["ct"], "waiting": [len(w) for w in st["waiting"]],
            "rush_waiting": sum(sum(w) for w in st["waiting"]),
            "labels": {k: v for k, v in labels.items() if v is not None}}


os.makedirs(args.out, exist_ok=True)
if args.preview:
    frames = []
    for i in range(len(SHOTS)):
        idx = [f for f, p in enumerate(PLAN) if p[0] == i]
        frames.append(idx[len(idx) // 2])
elif args.frames:
    frames = [int(v) for v in args.frames.split(",")]
else:
    frames = list(range(args.start, len(PLAN) if args.end < 0 else min(args.end, len(PLAN))))

hud = []
apply(frames[0])
for _ in range(4):                                     # warm-up: shaders, texture streaming
    rep.orchestrator.step(rt_subframes=24)
t0 = time.time()
for k, f in enumerate(frames):
    h = apply(f)
    hud.append(h)
    if args.hud_only:
        continue
    rep.orchestrator.step(rt_subframes=args.subframes)
    img = np.asarray(rgb.get_data())
    Image.fromarray(img[..., :3]).save(os.path.join(args.out, f"{f:04d}.png"))
    if k % 30 == 0 or args.preview:
        print(f"frame {f} ({k + 1}/{len(frames)}) {h['shot']} t={h['t']:.1f} "
              f"{(time.time() - t0) / (k + 1):.2f} s/frame", flush=True)
meta = {"W": W, "H": H, "fps": FPS, "policy": T["policy"], "n_robots": T["n_robots"], "nz": NZ,
        "shift": SHIFT, "n_frames": len(PLAN), "shots": [[s[0], s[1], s[2], s[3]] for s in SHOTS]}
name = "hud_preview.json" if (args.preview or args.frames) else f"hud_{args.start:04d}.json"
json.dump({"meta": meta, "frames": hud}, open(os.path.join(args.out, name), "w"))
print("DONE", len(frames), "frames", flush=True)
app.close()
