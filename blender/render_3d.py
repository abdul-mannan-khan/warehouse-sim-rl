"""3D replay of a simulated shift in Blender (Cycles, GPU).

    blender -b -P blender/render_3d.py -- --trace media/trace.json --out /tmp/frames/ --seconds 60
    blender -b -P blender/render_3d.py -- --trace media/trace.json --out /tmp/frames/ --preview   # quick 1-frame test

Everything is driven by the event trace: each frame, a handler computes where every robot is
(driving out, queueing at a picker, being picked, driving back), how many orders wait in each zone,
and updates the on-screen clock and figures. No keyframes are baked.
"""
import argparse
import json
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--out", default="/tmp/frames/")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--samples", type=int, default=48)
ap.add_argument("--res", default="1280x720")
ap.add_argument("--preview", action="store_true")
args = ap.parse_args(argv)

T = json.load(open(args.trace))
ORDERS = sorted(T["orders"], key=lambda o: o["created"])
NZ = len(T["zone_distance_m"])
SCALE = 0.25                      # metres in the model -> scene units (keeps the scene compact)
DEPOT = Vector((0.0, 0.0, 0.0))
ZONE_Y = [(z - (NZ - 1) / 2) * 6.0 for z in range(NZ)]
ZONES = [Vector((d * SCALE, ZONE_Y[z], 0.0)) for z, d in enumerate(T["zone_distance_m"])]
MAXQ = 16

# ---------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.cycles.samples = args.samples
scn.cycles.use_denoising = True
scn.render.use_persistent_data = True    # keep BVH/shaders between frames
w, h = map(int, args.res.split("x"))
scn.render.resolution_x, scn.render.resolution_y = w, h
scn.render.fps = args.fps
scn.frame_start, scn.frame_end = 1, int(args.seconds * args.fps)
scn.render.image_settings.file_format = "PNG"
scn.render.filepath = args.out
scn.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scn.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"

prefs = bpy.context.preferences.addons["cycles"].preferences
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == dev_type
            scn.cycles.device = "GPU"
            if dev_type == "OPTIX":
                scn.cycles.denoiser = "OPTIX"    # GPU denoiser; the CPU one is the bottleneck on small cloud VMs
            print("render device:", dev_type, [d.name for d in prefs.devices if d.use])
            break
    except TypeError:
        continue


def mat(name, rgb, emit=0.0, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        key = "Emission Color" if "Emission Color" in b.inputs else "Emission"
        b.inputs[key].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


M = {
    "floor": mat("floor", (0.55, 0.57, 0.6), rough=0.8),
    "line": mat("line", (0.95, 0.75, 0.1), rough=0.6),
    "rack": mat("rack", (0.12, 0.27, 0.55), rough=0.4, metal=0.6),
    "shelfbox": mat("shelfbox", (0.75, 0.55, 0.3), rough=0.9),
    "robot": mat("robot", (0.08, 0.08, 0.09), rough=0.35, metal=0.3),
    "normal": mat("normal", (0.15, 0.45, 0.95), rough=0.4),
    "rush": mat("rush", (0.95, 0.2, 0.15), rough=0.4, emit=0.6),
    "pick": mat("pick", (0.15, 0.85, 0.35), rough=0.4, emit=1.5),
    "idle": mat("idle", (0.4, 0.42, 0.45), rough=0.6),
    "person": mat("person", (1.0, 0.55, 0.1), rough=0.6),
    "depot": mat("depot", (0.85, 0.87, 0.9), rough=0.5),
    "text": mat("text", (1, 1, 1), emit=2.0),
    "textred": mat("textred", (1, 0.3, 0.25), emit=2.0),
}


def box(name, loc, size, material, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.data.materials.append(material)
    if parent:
        o.parent = parent
    return o


box("floor", (9, 0, -0.05), (38, 32, 0.1), M["floor"])
for z, p in enumerate(ZONES):                       # yellow floor lanes depot -> zone
    v = p - DEPOT
    ln = box(f"lane{z}", ((DEPOT + p) / 2) + Vector((0, 0, 0.005)), (v.length, 0.12, 0.01), M["line"])
    ln.rotation_euler[2] = math.atan2(v.y, v.x)
for z, p in enumerate(ZONES):                       # shelving racks with boxes either side of the pick face
    for side in (-1, 1):
        box(f"rack{z}{side}", p + Vector((1.6, side * 1.3, 1.0)), (3.0, 0.7, 2.0), M["rack"])
        for lvl in range(3):
            for k in range(3):
                box(f"sb{z}{side}{lvl}{k}", p + Vector((0.8 + k * 0.8, side * 1.3, 0.35 + lvl * 0.65)),
                    (0.55, 0.5, 0.4), M["shelfbox"])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=1.1, location=p + Vector((0.2, -0.75, 0.55)))
    bpy.context.object.data.materials.append(M["person"])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.17, location=p + Vector((0.2, -0.75, 1.28)))
    bpy.context.object.data.materials.append(M["person"])
box("depot", DEPOT + Vector((-1.6, 0, 0.4)), (2.0, 5.0, 0.8), M["depot"])

robots = []
for r in range(T["n_robots"]):
    body = box(f"robot{r}", (0, 0, 0.18), (0.9, 0.65, 0.28), M["robot"])
    tote = box(f"tote{r}", (0, 0, 0.32), (0.55, 0.45, 0.3), M["idle"], parent=body)
    tote.scale = (0.55 / 0.9, 0.45 / 0.65, 0.3 / 0.28)
    tote.location = (0, 0, 1.0)
    robots.append((body, tote))

queue_boxes = []                                   # waiting orders, stacked beside each zone
for z, p in enumerate(ZONES):
    row = []
    for i in range(MAXQ):
        b = box(f"q{z}_{i}", p + Vector((3.6 + (i % 8) * 0.42, -0.6 + (i // 8) * 0.45, 0.17)), (0.32, 0.32, 0.32), M["normal"])
        row.append(b)
    queue_boxes.append(row)

# lights, world, camera
bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
sun = bpy.context.object
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(-30))
bpy.ops.object.light_add(type="AREA", location=(8, 0, 12))
bpy.context.object.data.energy = 3000
bpy.context.object.data.size = 20
world = bpy.data.worlds.new("w")
scn.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.06, 0.08, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

bpy.ops.object.camera_add(location=(8.5, -27.0, 22.0))
cam = bpy.context.object
scn.camera = cam
cam.data.lens = 30
target = Vector((10.5, -0.8, 0.0))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


def text(name, body, loc, size, material):
    bpy.ops.object.text_add(location=(0, 0, 0))
    t = bpy.context.object
    t.name = name
    t.data.body = body
    t.data.size = size
    t.data.materials.append(material)
    t.parent = cam
    t.location = loc
    t.visible_shadow = False
    return t


# HUD text attached to the camera (z = -2 in front of lens); dark strip behind the bottom figures
hud_bar = box("hudbar", (0, 0, 0), (2.5, 0.15, 0.001), mat("hudbar", (0.02, 0.025, 0.035), rough=1.0), parent=cam)
hud_bar.location = (0.0, -0.603, -2.02)
hud_bar.rotation_euler = (0, 0, 0)
hud_bar.visible_shadow = False
hud_title = text("title", "SimPy warehouse simulator, 3D replay", (-1.12, 0.56, -2), 0.075, M["text"])
hud_sub = text("sub", f"Dispatch policy: {T['policy']}   |   {T['n_robots']} robots, {NZ} zones", (-1.12, 0.48, -2), 0.045, M["text"])
hud_clock = text("clock", "", (0.78, 0.56, -2), 0.075, M["text"])
hud_kpi = text("kpi", "", (-1.12, -0.6, -2), 0.05, M["text"])
hud_leg = text("legend", "blue tote = normal order   red = rush order   green = being picked   cubes = orders waiting",
               (-1.12, -0.66, -2), 0.035, M["text"])
for z, p in enumerate(ZONES):
    bpy.ops.object.text_add(location=p + Vector((0.3, 1.9, 2.3)))
    t = bpy.context.object
    t.data.body = f"Zone {z + 1}"
    t.data.size = 0.7
    t.data.materials.append(M["text"])
    t.rotation_euler = cam.rotation_euler
    t.visible_shadow = False
bpy.ops.object.text_add(location=DEPOT + Vector((-3.4, -3.6, 1.2)))
t = bpy.context.object
t.data.body = "Depot + packing"
t.data.size = 0.6
t.data.materials.append(M["text"])
t.rotation_euler = cam.rotation_euler
t.visible_shadow = False


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
            pos, heading, phase = DEPOT.lerp(zp, f), math.atan2(zp.y, zp.x), "out"
        elif o["pick_end"] is not None and t < o["pick_end"]:
            k = zone_slot[o["zone"]]
            zone_slot[o["zone"]] += 1
            pos = zp + Vector((-0.2 - 1.1 * k, 0.0, 0.0))
            heading = 0.0
            phase = "pick" if (o["pick_start"] is not None and t >= o["pick_start"]) else "queue"
        else:
            f = (t - o["pick_end"]) / max(o["back"] - o["pick_end"], 1e-9)
            pos, heading, phase = zp.lerp(DEPOT, f), math.atan2(-zp.y, -zp.x), "back"
        robot_state[o["robot"]] = (pos, heading, phase, o["rush"])
    return waiting, robot_state, done, late, (ct_sum / done if done else 0.0)


def on_frame(scene, *_):
    t = (scene.frame_current - 1) / (scene.frame_end - 1) * T["minutes"]
    waiting, rs, done, late, ct = state_at(t)
    for r, (body, tote) in enumerate(robots):
        if r in rs:
            pos, heading, phase, rush = rs[r]
            body.location = (pos.x, pos.y, 0.18)
            body.rotation_euler = (0, 0, heading)
            tote.material_slots[0].material = M["pick"] if phase == "pick" else (M["rush"] if rush else M["normal"])
        else:                                        # parked at the depot
            body.location = (DEPOT.x - 0.4, DEPOT.y - 2.0 + r * 0.8, 0.18)
            body.rotation_euler = (0, 0, 0)
            tote.material_slots[0].material = M["idle"]
    for z, row in enumerate(queue_boxes):
        for i, b in enumerate(row):
            show = i < len(waiting[z])
            b.hide_render = b.hide_viewport = not show
            if show:
                b.material_slots[0].material = M["rush"] if waiting[z][i] else M["normal"]
    hud_clock.data.body = f"t = {int(t) // 60:02d}:{int(t) % 60:02d}"
    hud_kpi.data.body = (f"done {done}    late {late}    mean cycle time {ct:4.1f} min    "
                         f"waiting {sum(len(w) for w in waiting)}")


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(on_frame)

if args.preview:
    scn.frame_set(int(scn.frame_end * 0.6))
    scn.render.filepath = args.out.rstrip("/") + "/preview.png"
    bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
