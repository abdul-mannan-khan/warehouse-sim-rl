"""Train a PPO dispatch policy and benchmark it against the heuristic baselines on held-out seeds.

    python train.py --timesteps 200000
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from warehouse_sim import SimConfig, WarehouseSim, WarehouseDispatchEnv
from warehouse_sim.policies import BASELINES

EVAL_SEEDS = range(500, 520)   # never used in training


def evaluate_policy_fn(policy_fn, cfg: SimConfig) -> dict:
    ms = [WarehouseSim(cfg.with_(seed=s), policy=policy_fn).run() for s in EVAL_SEEDS]
    return summarise(ms)


def evaluate_agent(model, cfg: SimConfig, masked: bool = False) -> dict:
    ms = []
    for s in EVAL_SEEDS:
        env = WarehouseDispatchEnv(cfg.with_(seed=s), randomise_seed=False)
        obs, _ = env.reset()
        while True:
            if masked:
                action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                ms.append(info)
                break
    return summarise(ms)


def summarise(ms: list[dict]) -> dict:
    keys = ["throughput_per_hour", "cycle_time_mean_min", "cycle_time_p95_min", "late_rate", "rush_late_rate", "invalid_actions"]
    return {k: (round(float(np.nanmean([m[k] for m in ms])), 3), round(float(np.nanstd([m[k] for m in ms])), 3)) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--masked", action="store_true", help="MaskablePPO (sb3-contrib): never choose an empty zone")
    args = ap.parse_args()

    from stable_baselines3.common.env_util import make_vec_env

    cfg = SimConfig()
    venv = make_vec_env(lambda: WarehouseDispatchEnv(cfg), n_envs=4, seed=args.seed)
    kw = dict(n_steps=512, batch_size=256, gamma=0.995, ent_coef=0.01, verbose=0, seed=args.seed)
    if args.masked:
        from sb3_contrib import MaskablePPO   # uses env.action_masks() automatically
        model = MaskablePPO("MlpPolicy", venv, **kw)
    else:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", venv, **kw)
    model.learn(total_timesteps=args.timesteps)
    name = "maskable_ppo_dispatch" if args.masked else "ppo_dispatch"
    model.save(name)

    results = {name: evaluate_policy_fn(fn, cfg) for name, fn in BASELINES.items()}
    results[name] = evaluate_agent(model, cfg, masked=args.masked)
    print("mean, std over", len(EVAL_SEEDS), "held-out shifts")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
