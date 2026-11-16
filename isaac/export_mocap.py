"""Convert Adobe Mixamo FBX clips to USD and dump their skeleton and joint rotations for replay_isaac.py.

    /isaac-sim/python.sh isaac/export_mocap.py --fbx mocap/fbx --out mocap/

Writes mocap/clip_<name>.npz (source skeleton, bind pose, per-frame joint rotations and translations).
The replay expects box_turn_b, putting_down and picking_up (see the README for the exact Mixamo clips).
"""
import argparse
import asyncio
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("--fbx", default="fbx")
ap.add_argument("--out", default="dump")
args = ap.parse_args()

from isaacsim import SimulationApp                      # noqa: E402
app = SimulationApp({"headless": True})

import numpy as np                                     # noqa: E402
from pxr import Usd, UsdGeom, UsdSkel                  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()
import omni.kit.asset_converter as ac                  # noqa: E402
from retarget import mat_rot, parents, leaf             # noqa: E402

os.makedirs(args.out, exist_ok=True)


def run(coro):
    fut = asyncio.ensure_future(coro)
    while not fut.done():
        app.update()
    return fut.result()


# ---------------------------------------------------------------- FBX -> USD
for fbx in sorted(glob.glob(os.path.join(args.fbx, "*.fbx"))):
    name = os.path.splitext(os.path.basename(fbx))[0]
    usd = os.path.join(args.out, f"{name}.usd")
    ctx = ac.AssetConverterContext()
    ctx.ignore_materials = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.use_meter_as_world_unit = True
    task = ac.get_instance().create_converter_task(os.path.abspath(fbx), os.path.abspath(usd), None, ctx)
    ok = run(task.wait_until_finished())
    print("convert", name, ok, task.get_status(), task.get_error_message() if not ok else "", flush=True)
    st = Usd.Stage.Open(usd)
    sk = next((p for p in st.Traverse() if p.IsA(UsdSkel.Skeleton)), None)
    an = next((p for p in st.Traverse() if p.IsA(UsdSkel.Animation)), None)
    print("  skel", sk.GetPath() if sk else None, "anim", an.GetPath() if an else None,
          "tcps", st.GetTimeCodesPerSecond(), "range", st.GetStartTimeCode(), st.GetEndTimeCode(),
          "metersPerUnit", UsdGeom.GetStageMetersPerUnit(st), "up", UsdGeom.GetStageUpAxis(st), flush=True)
    if sk is None or an is None:
        continue
    sk, an = UsdSkel.Skeleton(sk), UsdSkel.Animation(an)
    sj = list(sk.GetJointsAttr().Get())
    bind = [mat_rot(m) for m in sk.GetBindTransformsAttr().Get()]
    rest = [mat_rot(m) for m in sk.GetRestTransformsAttr().Get()] if sk.GetRestTransformsAttr().HasValue() else bind
    aj = list(an.GetJointsAttr().Get())
    ts = an.GetRotationsAttr().GetTimeSamples() or [0.0]
    R, Tt = [], []
    for t in ts:
        R.append([[q.GetReal(), *q.GetImaginary()] for q in an.GetRotationsAttr().Get(t)])
        tr = an.GetTranslationsAttr().Get(t)
        Tt.append([list(v) for v in tr] if tr is not None else [[0, 0, 0]] * len(aj))
    # skeleton prim world transform (unit scale and axis conversion live here)
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(sk.GetPrim())
    np.savez(os.path.join(args.out, f"clip_{name}.npz"),
             sj=np.array([leaf(j) for j in sj]), spar=np.array(parents(sj)),
             bind_r=np.stack([r for r, _ in bind]), bind_t=np.stack([t for _, t in bind]),
             rest_r=np.stack([r for r, _ in rest]), rest_t=np.stack([t for _, t in rest]),
             aj=np.array([leaf(j) for j in aj]), apar=np.array(parents(aj)),
             rot=np.array(R, float), trans=np.array(Tt, float), times=np.array(ts, float),
             tcps=st.GetTimeCodesPerSecond(), xf=np.array([[xf[i][j] for j in range(4)] for i in range(4)]),
             mpu=UsdGeom.GetStageMetersPerUnit(st))
    print("  joints", len(sj), "anim joints", len(aj), "frames", len(ts), "first", [leaf(j) for j in sj[:4]], flush=True)


print("EXPORT DONE", flush=True)
app.close()
