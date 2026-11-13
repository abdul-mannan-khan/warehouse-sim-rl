"""3D replay of a simulated shift in NVIDIA Isaac Sim (headless, RTX).

    /isaac-sim/python.sh isaac/replay_isaac.py --trace media/trace.json --out frames/ --seconds 60
    /isaac-sim/python.sh isaac/replay_isaac.py --trace media/trace.json --out frames/ --preview

Runs on a rented Linux RTX GPU (Isaac Sim 6.2.0 container); assets come from NVIDIA's public asset server.
The SimPy event trace drives the scene: each frame we compute where every robot is (driving out,
queueing at a picker, being picked, driving back) and how many orders wait per zone, move the
robots (NVIDIA iw.hub warehouse AMRs) kinematically and render with the RTX renderer through Replicator.
Frames are raw RGB; overlay.py adds the clock, KPIs and labels (positions saved in hud.json).
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--out", default="frames/")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--subframes", type=int, default=8)
ap.add_argument("--preview", action="store_true")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

from isaacsim import SimulationApp                      # noqa: E402
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": W, "height": H})

import numpy as np                                     # noqa: E402
import omni.client                                     # noqa: E402
import omni.replicator.core as rep                     # noqa: E402
import omni.usd                                        # noqa: E402
from PIL import Image                                  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

ROOT = get_assets_root_path()
print("assets:", ROOT, flush=True)

T = json.load(open(args.trace))
ORDERS = sorted(T["orders"], key=lambda o: o["created"])
NZ = len(T["zone_distance_m"])
# Layout in metres. Model distances (20-80 m) are compressed so the whole floor fits one view.
DEPOT = np.array([0.0, 0.0])
ZONE_Y = [(z - (NZ - 1) / 2) * 7.0 for z in range(NZ)]
ZONES = [np.array([6.0 + d * 0.3, ZONE_Y[z]]) for z, d in enumerate(T["zone_distance_m"])]
MAXQ = 16

omni.usd.get_context().new_stage()
for _ in range(5):
    app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def exists(url):
    r, _ = omni.client.stat(url)
    return r == omni.client.Result.OK


def first(*rel):
    for r in rel:
        if exists(f"{ROOT}/{r}"):
            return f"{ROOT}/{r}"
    print("asset not found, using primitives:", rel[0], flush=True)
    return None


A_ROBOT = first("Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd", "Isaac/Robots/Idealworks/iwhub/iw_hub.usd",
                "Isaac/Robots/Idealworks/iw_hub.usd", "Isaac/Robots/Idealworks/iw_hub_static.usd",
                "Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd", "Isaac/Robots/Carter/nova_carter.usd")
A_PERSON = first("Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
                 "Isaac/People/Characters/F_Business_02/F_Business_02.usd")
A_BOX = first("Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd")

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


def xform(path, pos=(0, 0, 0), yaw=0.0, scale=None):
    x = UsdGeom.Xform.Define(stage, path)
    t = x.AddTranslateOp()
    r = x.AddRotateZOp()
    t.Set(Gf.Vec3d(*pos))
    r.Set(float(yaw))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*scale))
    return x, t, r


def cube(path, pos, size, m):
    x, t, r = xform(path, pos, 0.0, size)
    c = UsdGeom.Cube.Define(stage, path + "/geo")
    c.CreateSizeAttr(1.0)
    bind(c.GetPrim(), m)
    return x, t


def reference(path, url, pos=(0, 0, 0), yaw=0.0, scale=None):
    x, t, r = xform(path, pos, yaw, scale)
    child = stage.DefinePrim(path + "/asset", "Xform")
    child.GetReferences().AddReference(url)
    return x, t, r


def bbox(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


M = {
    "floor": material("floor", (0.26, 0.27, 0.29), rough=0.7),
    "line": material("line", (0.95, 0.72, 0.05), rough=0.6),
    "rack": material("rack", (0.12, 0.27, 0.55), rough=0.4, metal=0.6),
    "shelfbox": material("shelfbox", (0.72, 0.52, 0.3), rough=0.9),
    "normal": material("normal", (0.1, 0.35, 0.95), rough=0.4, emit=0.12),
    "rush": material("rush", (0.95, 0.15, 0.1), rough=0.4, emit=1.2),
    "pick": material("pick", (0.1, 0.9, 0.3), rough=0.4, emit=2.0),
    "idle": material("idle", (0.45, 0.47, 0.5), rough=0.6),
    "depot": material("depot", (0.82, 0.84, 0.88), rough=0.5),
    "person": material("person", (1.0, 0.55, 0.1), rough=0.6),
}

# ---------------------------------------------------------------- static scene
cube("/World/floor", (14.0, 0.0, -0.05), (120.0, 100.0, 0.1), M["floor"])
for z, p in enumerate(ZONES):                       # yellow floor lanes depot -> zone
    v = p - DEPOT
    mid = (DEPOT + p) / 2
    x, _, r = xform(f"/World/lane{z}", (mid[0], mid[1], 0.003), math.degrees(math.atan2(v[1], v[0])))
    c = UsdGeom.Cube.Define(stage, f"/World/lane{z}/geo")
    c.CreateSizeAttr(1.0)
    c.AddScaleOp().Set(Gf.Vec3f(float(np.linalg.norm(v)), 0.15, 0.004))
    bind(c.GetPrim(), M["line"])

M["upright"] = material("upright", (0.08, 0.22, 0.6), rough=0.35, metal=0.5)
M["beam"] = material("beam", (0.95, 0.42, 0.05), rough=0.4, metal=0.4)
M["deck"] = material("deck", (0.62, 0.64, 0.66), rough=0.6, metal=0.1)
BOXES = []
for z, p in enumerate(ZONES):                       # pallet racking either side of each pick face, picker in front
    for side in (-1, 1):
        rx, ry = p[0] + 2.4, p[1] + side * 1.7
        base = f"/World/rack{z}_{side + 1}"
        UsdGeom.Xform.Define(stage, base)
        for ix, dx in enumerate((-2.0, 0.0, 2.0)):              # 3 upright frames -> 2 bays of 2 m
            for iy, dy in enumerate((-0.5, 0.5)):
                cube(f"{base}/up{ix}{iy}", (rx + dx, ry + dy, 1.2), (0.09, 0.09, 2.4), M["upright"])
        for lvl, hz in enumerate((0.15, 0.95, 1.75)):
            for dy in (-0.5, 0.5):
                cube(f"{base}/beam{lvl}{int(dy > 0)}", (rx, ry + dy, hz), (4.0, 0.08, 0.12), M["beam"])
            cube(f"{base}/deck{lvl}", (rx, ry, hz + 0.07), (4.0, 1.0, 0.02), M["deck"])
            for k in range(4):
                if A_BOX and (z + lvl + k + side) % 5 != 0:      # leave a few gaps, like a real pick face
                    path = f"{base}/box{lvl}{k}"
                    reference(path, A_BOX, (rx - 1.5 + k * 1.0, ry, hz + 0.08), 90.0 * ((k + lvl) % 2))
                    BOXES.append(path)
                elif not A_BOX:
                    cube(f"{base}/box{lvl}{k}", (rx - 1.5 + k * 1.0, ry, hz + 0.33), (0.7, 0.6, 0.5), M["shelfbox"])
    if A_PERSON:
        reference(f"/World/picker{z}", A_PERSON, (p[0] + 0.3, p[1] - 0.95, 0.0), 90.0)
    else:
        cube(f"/World/picker{z}", (p[0] + 0.3, p[1] - 0.95, 0.85), (0.4, 0.4, 1.7), M["person"])
cube("/World/depot", (DEPOT[0] - 2.4, DEPOT[1], 0.45), (1.6, 8.0, 0.9), M["depot"])      # packing benches
M["roller"] = material("roller", (0.3, 0.32, 0.35), rough=0.3, metal=0.8)
cube("/World/conveyor", (DEPOT[0] - 4.2, DEPOT[1], 0.4), (1.0, 9.0, 0.8), M["roller"])
for i, dy in enumerate((-2.6, 0.0, 2.6)):
    if A_BOX:
        reference(f"/World/pbox{i}", A_BOX, (DEPOT[0] - 4.2, DEPOT[1] + dy + 0.9, 0.8), 15.0 * i)
        BOXES.append(f"/World/pbox{i}")

robots = []
for r in range(T["n_robots"]):
    if A_ROBOT:
        x, t, rot = reference(f"/World/robot{r}", A_ROBOT)
    else:
        x, t, rot = xform(f"/World/robot{r}")
        c = UsdGeom.Cube.Define(stage, f"/World/robot{r}/geo")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.18))
        c.AddScaleOp().Set(Gf.Vec3f(1.2, 0.8, 0.3))
        bind(c.GetPrim(), material("robotbody", (0.08, 0.08, 0.09), rough=0.35, metal=0.3))
    robots.append([x, t, rot, None])

queue = []                                          # waiting orders, stacked beside each zone
for z, p in enumerate(ZONES):
    row = []
    for i in range(MAXQ):
        x, t = cube(f"/World/q{z}_{i}", (p[0] + 4.9 + (i % 8) * 0.5, p[1] - 0.5 + (i // 8) * 0.55, 0.2),
                    (0.4, 0.4, 0.4), M["normal"])
        row.append(x)
    queue.append(row)

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(float(os.environ.get("DOME", 250)))
dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(float(os.environ.get("SUN", 900)))
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 20.0, -30.0))

# let referenced assets stream in from the public asset server before measuring them
for _ in range(60):
    app.update()


def fit(path, length, along_x=True):
    """Scale a referenced asset (some warehouse props are authored in cm) to `length` m and turn it lengthwise."""
    child = stage.GetPrimAtPath(path + "/asset")
    s = bbox(child).GetSize()
    if max(s[0], s[1]) <= 0:
        return
    k = length / max(s[0], s[1])
    UsdGeom.Xformable(child).AddScaleOp(opSuffix="fit").Set(Gf.Vec3f(k, k, k))
    if along_x and s[1] > s[0]:
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[1].Set(90.0)
    g = bbox(child)
    parent_t = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0].Get()
    print(f"fit {path}: scale {k:.3f} size {g.GetSize()} min {g.GetMin()} max {g.GetMax()} target {parent_t}", flush=True)


def fit_box(path, size):
    """Scale a referenced prop (warehouse props are authored in cm) so its largest side is `size` m, base on z."""
    child = stage.GetPrimAtPath(path + "/asset")
    g = bbox(child)
    s = g.GetSize()
    if max(s) <= 0:
        return
    k = size / max(s)
    UsdGeom.Xformable(child).AddScaleOp(opSuffix="fit").Set(Gf.Vec3f(k, k, k))
    g2 = bbox(child)
    top = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0]
    t = top.Get()
    top.Set(Gf.Vec3d(t[0], t[1], t[2] + (t[2] - g2.GetMin()[2])))   # stand the box on its shelf


for i, b in enumerate(BOXES):
    fit_box(b, 0.6)
if BOXES:
    print("box", bbox(stage.GetPrimAtPath(BOXES[0])).GetMin(), bbox(stage.GetPrimAtPath(BOXES[0])).GetMax(), flush=True)


# robot size -> tote height; fit any robot asset to about 1.3 m long
ROBOT_TOP = 0.36
if A_ROBOT:
    rg = bbox(stage.GetPrimAtPath("/World/robot0"))
    size = rg.GetSize()
    print("robot bbox", size, flush=True)
    if size[0] > 0:
        ROBOT_TOP = float(rg.GetMax()[2])
for r, rb in enumerate(robots):
    x, t = cube(f"/World/robot{r}/tote", (0.0, 0.0, ROBOT_TOP + 0.17), (0.6, 0.45, 0.34), M["idle"])
    rb[3] = x.GetPrim().GetChild("geo")

CAM_EYE = Gf.Vec3d(13.5, -31.0, 34.0)
CAM_TGT = Gf.Vec3d(15.5, -1.5, 0.0)
FOCAL, HAPER = 18.0, 20.955
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(FOCAL)
cam.CreateHorizontalApertureAttr(HAPER)
cam.CreateVerticalApertureAttr(HAPER * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 500.0))
VIEW = Gf.Matrix4d().SetLookAt(CAM_EYE, CAM_TGT, Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(VIEW.GetInverse())


def project(p):
    c = VIEW.Transform(Gf.Vec3d(*p))
    k = FOCAL / HAPER * W
    return [W / 2 + k * c[0] / -c[2], H / 2 - k * c[1] / -c[2]]


rp = rep.create.render_product("/World/cam", (W, H))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])


# ---------------------------------------------------------------- per-frame state from the trace
def state_at(t):
    waiting = [[] for _ in range(NZ)]
    robot_state = {}
    done = late = 0
    ct_sum = 0.0
    zone_slot = [0] * NZ
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
        zp = ZONES[o["zone"]]
        if o["at_zone"] is not None and t < o["at_zone"]:
            f = (t - o["dispatched"]) / max(o["at_zone"] - o["dispatched"], 1e-9)
            pos, heading, phase = DEPOT + (zp - DEPOT) * f, math.atan2(zp[1], zp[0]), "out"
        elif o["pick_end"] is None or t < o["pick_end"]:      # still at the zone when the shift ends
            k = zone_slot[o["zone"]]
            zone_slot[o["zone"]] += 1
            pos, heading = zp + np.array([-0.3 - 1.5 * k, 0.0]), 0.0
            phase = "pick" if (o["pick_start"] is not None and t >= o["pick_start"]) else "queue"
        else:
            back = o["back"] if o["back"] is not None else o["pick_end"] + (o["at_zone"] - o["dispatched"])
            f = min((t - o["pick_end"]) / max(back - o["pick_end"], 1e-9), 1.0)
            pos, heading, phase = zp + (DEPOT - zp) * f, math.atan2(-zp[1], -zp[0]), "back"
        robot_state[o["robot"]] = (pos, heading, phase, o["rush"])
    return waiting, robot_state, done, late, (ct_sum / done if done else 0.0)


def apply(t):
    waiting, rs, done, late, ct = state_at(t)
    for r, (x, tr, rot, tote) in enumerate(robots):
        if r in rs:
            pos, heading, phase, rush = rs[r]
            tr.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.0))
            rot.Set(math.degrees(heading))
            bind(tote, M["pick"] if phase == "pick" else (M["rush"] if rush else M["normal"]))
        else:                                        # parked at the depot
            tr.Set(Gf.Vec3d(DEPOT[0] - 0.2, DEPOT[1] - 3.0 + r * 1.2, 0.0))
            rot.Set(0.0)
            bind(tote, M["idle"])
    for z, row in enumerate(queue):
        for i, x in enumerate(row):
            img = UsdGeom.Imageable(x.GetPrim())
            if i < len(waiting[z]):
                img.MakeVisible()
                bind(x.GetPrim().GetChild("geo"), M["rush"] if waiting[z][i] else M["normal"])
            else:
                img.MakeInvisible()
    return {"t": t, "done": done, "late": late, "ct": ct, "waiting": sum(len(w) for w in waiting)}


os.makedirs(args.out, exist_ok=True)
n = int(args.seconds * args.fps)
frames = [int(n * 0.6)] if args.preview else list(range(args.start, n if args.end < 0 else min(args.end, n)))
labels = {"zones": [project((p[0] + 5.0, p[1] + 1.4, 0.0)) for p in ZONES],
          "depot": project((DEPOT[0] - 4.0, DEPOT[1] - 4.6, 0.0)), "W": W, "H": H,
          "policy": T["policy"], "n_robots": T["n_robots"], "nz": NZ}
hud = []
apply(0.0)
for _ in range(3):                                   # warm-up: shader compile, texture streaming
    rep.orchestrator.step(rt_subframes=16)
import time                                           # noqa: E402
t0 = time.time()
for k, f in enumerate(frames):
    t = f / max(n - 1, 1) * T["minutes"]
    h = apply(t)
    h["frame"] = f
    hud.append(h)
    rep.orchestrator.step(rt_subframes=args.subframes)
    img = np.asarray(rgb.get_data())
    Image.fromarray(img[..., :3]).save(os.path.join(args.out, ("preview" if args.preview else f"{f:04d}") + ".png"))
    if k % 30 == 0:
        print(f"frame {f} ({k + 1}/{len(frames)}) {(time.time() - t0) / (k + 1):.2f} s/frame", flush=True)
json.dump({"labels": labels, "frames": hud}, open(os.path.join(args.out, "hud.json" if not args.preview
                                                              else "hud_preview.json"), "w"))
print("DONE", len(frames), "frames", flush=True)
app.close()
