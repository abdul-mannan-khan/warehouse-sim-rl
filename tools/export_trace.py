"""Export one simulated shift as a JSON event trace (for the 3D Blender replay in blender/render_3d.py).

    python tools/export_trace.py --policy least_loaded_zone --minutes 120 --out media/trace.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warehouse_sim import SimConfig, WarehouseSim, WarehouseDispatchEnv  # noqa: E402
from warehouse_sim.policies import BASELINES  # noqa: E402


def assign_robot_ids(orders, n_robots):
    """Give each trip a robot id: a robot is busy from dispatch until it is back at the depot."""
    free_at = [0.0] * n_robots
    for o in sorted((o for o in orders if o.dispatched is not None), key=lambda o: o.dispatched):
        rid = min(range(n_robots), key=lambda r: (free_at[r] > o.dispatched + 1e-9, free_at[r]))
        o.robot = rid
        free_at[rid] = o.back if o.back is not None else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="least_loaded_zone", help=f"one of {list(BASELINES)} or a .zip model")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="media/trace.json")
    args = ap.parse_args()

    cfg = SimConfig(seed=args.seed, horizon_min=args.minutes)
    if args.policy.endswith(".zip"):
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.policy)
        env = WarehouseDispatchEnv(cfg, randomise_seed=False)
        obs, _ = env.reset()
        while True:
            a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                sim = env.sim
                break
        label = "Masked PPO (learned)"
    else:
        sim = WarehouseSim(cfg, policy=BASELINES[args.policy])
        sim.run()
        label = args.policy.replace("_", "-")

    assign_robot_ids(sim.orders, cfg.n_robots)
    keys = ["id", "created", "zone", "lines", "rush", "dispatched", "at_zone", "pick_start", "pick_end", "back", "done"]
    trace = {
        "policy": label,
        "minutes": args.minutes,
        "n_robots": cfg.n_robots,
        "zone_distance_m": list(cfg.zone_distance_m),
        "sla_rush_min": cfg.sla_rush_min,
        "sla_normal_min": cfg.sla_normal_min,
        "orders": [dict({k: getattr(o, k) for k in keys}, robot=getattr(o, "robot", None)) for o in sim.orders],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(trace, f)
    print(f"wrote {args.out}: {len(trace['orders'])} orders, policy {label}, metrics {sim.metrics()}")


if __name__ == "__main__":
    main()
