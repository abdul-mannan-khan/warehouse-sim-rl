"""Replay a simulated shift inside NVIDIA's warehouse digital twin (Isaac Sim 6.2, headless, RTX).

    /isaac-sim/python.sh isaac/replay_isaac.py --trace media/trace.json --out frames/
    /isaac-sim/python.sh isaac/replay_isaac.py --trace media/trace.json --out preview/ --preview

The SimPy event trace drives everything. For each frame we compute where every order is in its life
cycle (waiting, robot driving out, queueing at the picker, being picked, driving back) and place NVIDIA's
iw.hub robots along aisle routes in the `full_warehouse` environment. The four picking zones of the model are
mapped to four rack aisles and the depot to a packing area on the open floor. Pickers and packers are
NVIDIA's animated characters; a picker plays a picking gesture while the simulator says a pick is in progress.

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
        "pick": (f"{ROOT}/Isaac/People/Animations/type_keyboard.skelanim.usd", None),       # bends to the tote
        "pack": (f"{ROOT}/Isaac/People/MotionLibrary/BuiltinActions/MoveCarryObject/CarryObjectIdleBothHands.usd", 24.0)}

T = json.load(open(args.trace))
ORDERS = sorted(T["orders"], key=lambda o: o["created"])
NZ = len(T["zone_distance_m"])
SHIFT = T["minutes"]

# ------------------------------------------------------------------ layout (metres, warehouse frame)
# Racks run along +y between y = 8.8 and 24.9; aisle centre lines sit between rack rows.
AISLE_X = [-22.85, -17.9, -12.95, -8.0, -3.0, 1.95]
ZONE_AISLE = [AISLE_X[3], AISLE_X[2], AISLE_X[1], AISLE_X[0]]      # the four clear aisles; zone 1 nearest
ZONE_PICK_Y = [12.5, 15.5, 18.5, 21.5]
LANE = 0.6                       # out lane on the +x side of an aisle, return lane on the -x side
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
    picking = [False] * NZ
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
            picking[z] |= phase == "pick"
        else:
            back = o["back"] if o["back"] is not None else o["pick_end"] + (o["at_zone"] - o["dispatched"])
            f = (t - o["pick_end"]) / max(back - o["pick_end"], 1e-9)
            pos, heading = o["P_back"].pose(o["P_back"].L * ease(f))
            phase = "back"
        robots[o["robot"]] = (pos, heading, phase, o["rush"], z)
    return dict(waiting=waiting, robots=robots, picking=picking, done=done, late=late,
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
AISLE_ZONE = 1
a0 = busiest(SHOTS[-1][3] + 1, 2.0, lambda a, b: sum(1 for o in ORDERS if o["zone"] == AISLE_ZONE and o["pick_start"]
                                                     and a + 0.3 < o["pick_start"] < b - 0.3), t_to=200)
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
PACK_Y = DOCK_Y - 3.0
packers = []
for i, x in enumerate((DOCK_X[0] + 0.9, (DOCK_X[0] + DOCK_X[-1]) / 2, DOCK_X[-1] - 0.9)):
    reference(f"/World/pack/table{i}", A_TABLE, (x, PACK_Y, 0.0), 0.0)
    packers.append((f"/World/pack/packer{i}", (x, PACK_Y - 1.1), 90.0))     # facing the docks
for i, (x, y) in enumerate([(-23.0, -5.0), (-23.0, -7.2), (-21.0, -5.0), (-21.0, -7.5), (-1.5, -9.0), (0.5, -9.0)]):
    reference(f"/World/pal/p{i}", A_PALLET, (x, y, 0.0), 90.0 * (i % 2))
    for j in range(3 if i % 3 else 2):
        reference(f"/World/pal/p{i}b{j}", A_BOX, (x - 0.3 + 0.6 * (j % 2), y + 0.25 - 0.5 * (j // 2), 0.16), 0.0)
reference("/World/forklift", A_FORKLIFT, (-22.5, -11.5, 0.0), 35.0)

# pickers: one per zone, standing at the pick face beside the robot stop
pickers = [(f"/World/picker{z}", (ZONE_AISLE[z] + 1.2, ZONE_PICK_Y[z] + 0.1), 180.0) for z in range(NZ)]
CHARS, CHAR_URL = [], []
for k, (path, (x, y), heading) in enumerate(pickers + packers):
    reference(path, PEOPLE[k % len(PEOPLE)], (x, y, 0.0), heading + 90.0)   # characters face -y by default
    CHARS.append(path)
    CHAR_URL.append(PEOPLE[k % len(PEOPLE)])

robots = []
for r in range(T["n_robots"]):
    x, tr, rot = reference(f"/World/robot{r}", A_ROBOT, (DOCK_X[r], DOCK_Y, 0.0), -90.0 + YAW_OFF)
    robots.append({"x": x, "t": tr, "r": rot})

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
for p in stage.GetPrimAtPath("/World/pal").GetChildren():
    fit(str(p.GetPath()), 1.2 if "b" not in p.GetName() else 0.5)
fit("/World/forklift", 3.2)
for c in CHARS:
    fit(c, 1.76, axis="z")
rb = bbox(stage.GetPrimAtPath("/World/robot0/asset"))
print("robot bbox", rb.GetSize(), rb.GetMin(), rb.GetMax(), flush=True)
ROBOT_TOP = float(rb.GetMax()[2]) if rb.GetSize()[0] > 0 else 0.35
for r, rbt in enumerate(robots):
    path = f"/World/robot{r}/tote"
    reference(path, A_TOTE, (0.0, 0.0, ROBOT_TOP + 0.01), 90.0 - YAW_OFF)
    fit(path, 0.6)
    tote_geo = stage.GetPrimAtPath(path)
    bind(tote_geo, M["idle"])
    rbt["tote"] = tote_geo


# ------------------------------------------------------------------ character animation
# NVIDIA's clips target the biped rig; the characters use a Reallusion rig, so retarget them (retarget.py).
import sys                                             # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import Retargeter                        # noqa: E402

BIPED = f"{ROOT}/Isaac/People/Characters/Biped_Setup.usd"
RT, CLIPS, ANIMS = {}, {}, []
for i, c in enumerate(CHARS):
    skel = next((p for p in Usd.PrimRange(stage.GetPrimAtPath(c)) if p.IsA(UsdSkel.Skeleton)), None)
    url = CHAR_URL[i]
    if skel is None:
        ANIMS.append(None)
        continue
    try:
        if url not in RT:
            RT[url] = Retargeter(BIPED, skel)
            for k, (clip_url, max_s) in ANIM.items():
                CLIPS[(url, k)] = RT[url].clip(clip_url, FPS, max_s)
            print("retargeted", url.rsplit("/", 1)[-1], len(RT[url].tmap), "joints", flush=True)
    except Exception as e:                     # noqa: BLE001
        print("animation unavailable for", url, e, flush=True)
        ANIMS.append(None)
        continue
    r = RT[url]
    a = UsdSkel.Animation.Define(stage, f"/World/anim/a{i}")
    a.GetJointsAttr().Set(r.tjoints)
    a.GetTranslationsAttr().Set(r.t_trans)
    a.GetScalesAttr().Set(r.t_scale)
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([a.GetPath()])
    ANIMS.append({"a": a, "url": url, "phase": int(FPS * 1.37 * i)})
print("animated characters:", sum(a is not None for a in ANIMS), flush=True)


def animate(i, clip_name, sec):
    A = ANIMS[i]
    if A is None:
        return
    frames = CLIPS[(A["url"], clip_name)]
    A["a"].GetRotationsAttr().Set(frames[(int(sec * FPS) + A["phase"]) % len(frames)])


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
        eye = lerp((ax - 1.4, py + 4.5, 2.1), (ax - 1.0, py + 3.3, 1.8), k)
        return eye, (ax + 0.6, py - 5.0, 0.7), 16.0
    if name == "depot":
        k = smooth(u)
        eye = lerp((DOCK_X[-1] + 6.0, DOCK_Y + 7.0, 4.5), (DOCK_X[-1] + 3.5, DOCK_Y + 8.0, 4.2), k)
        return eye, ((DOCK_X[0] + DOCK_X[-1]) / 2 + 1.0, DOCK_Y - 1.5, 0.5), 18.0
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
            pos, heading, phase, rush, z = st["robots"][r]
            R["t"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.0))
            R["r"].Set(math.degrees(heading) + YAW_OFF)
            bind(R["tote"], M["pick"] if phase == "pick" else (M["rush"] if rush else M["normal"]))
        else:
            R["t"].Set(Gf.Vec3d(DOCK_X[r], DOCK_Y, 0.0))
            R["r"].Set(-90.0 + YAW_OFF)
            bind(R["tote"], M["idle"])
    sec = f / FPS
    docked = sum(1 for r in range(len(robots)) if r not in st["robots"])
    for i in range(len(CHARS)):
        if i < NZ:
            animate(i, "pick" if st["picking"][i] else "idle", sec)
        else:
            animate(i, "pack" if docked else "idle", sec)
    eye, tgt, focal = camera_for(f)
    view = set_camera(eye, tgt, focal)
    name = SHOTS[si][0]
    rate = (SHOTS[si][3] - SHOTS[si][2]) * 60.0 / SHOTS[si][1]
    labels = {}
    if name in ("establish", "overview"):
        labels = {f"Zone {z + 1}": project(view, focal, (ZONE_AISLE[z], 9.0, 6.6)) for z in range(NZ)}
        labels["Packing + docks"] = project(view, focal, ((DOCK_X[0] + DOCK_X[-1]) / 2, PACK_Y - 2.2, 0.0))
    elif name == "aisle":
        labels = {f"Zone {ex_z + 1} picker": project(view, focal, (ZONE_AISLE[ex_z] + 1.2, ZONE_PICK_Y[ex_z], 2.1))
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
                   "aisle": f"Zone {ex.get('zone', 0) + 1}: robots queue for the picker",
                   "depot": "Packing area: robots return totes and wait at their docks",
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
    rep.orchestrator.step(rt_subframes=args.subframes)
    img = np.asarray(rgb.get_data())
    Image.fromarray(img[..., :3]).save(os.path.join(args.out, f"{f:04d}.png"))
    if k % 30 == 0 or args.preview:
        print(f"frame {f} ({k + 1}/{len(frames)}) {h['shot']} t={h['t']:.1f} "
              f"{(time.time() - t0) / (k + 1):.2f} s/frame", flush=True)
meta = {"W": W, "H": H, "fps": FPS, "policy": T["policy"], "n_robots": T["n_robots"], "nz": NZ,
        "shift": SHIFT, "n_frames": len(PLAN), "shots": [[s[0], s[1], s[2], s[3]] for s in SHOTS]}
name = "hud_preview.json" if args.preview else f"hud_{args.start:04d}.json"
json.dump({"meta": meta, "frames": hud}, open(os.path.join(args.out, name), "w"))
print("DONE", len(frames), "frames", flush=True)
app.close()
