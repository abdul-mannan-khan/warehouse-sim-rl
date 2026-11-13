"""Close the sim-to-real gap with Optuna.

We pretend a real site exists: its "true" parameters are hidden, and all we see is a log of
order cycle times and hourly throughput (in practice this would come from warehouse operational data).
Optuna then searches simulator parameters so that the simulated distributions match the observed ones.

    python calibrate.py --trials 60
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import optuna

from warehouse_sim import SimConfig, WarehouseSim
from warehouse_sim.policies import fifo
from warehouse_sim.tracking import Tracker

TUNED = ("pick_time_mean_min", "robot_speed_m_per_min", "congestion")
HIDDEN_TRUTH = dict(pick_time_mean_min=0.62, robot_speed_m_per_min=48.0, congestion=0.09)


def simulate(cfg: SimConfig, seeds) -> tuple[np.ndarray, float]:
    cts, thr = [], []
    for s in seeds:
        sim = WarehouseSim(cfg.with_(seed=int(s)), policy=fifo)
        m = sim.run()
        cts.append(sim.cycle_times())
        thr.append(m["throughput_per_hour"])
    return np.concatenate(cts), float(np.mean(thr))


def wasserstein_1d(a: np.ndarray, b: np.ndarray, n_q: int = 200) -> float:
    """1-D Wasserstein distance via matched quantiles (no SciPy needed)."""
    q = np.linspace(0.005, 0.995, n_q)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def make_real_site_log(base: SimConfig, n_shifts: int = 6, noise_sd: float = 0.3):
    """'Observed' data from the hidden real site, with measurement noise on timestamps."""
    rng = np.random.default_rng(12345)
    ct, thr = simulate(base.with_(**HIDDEN_TRUTH), seeds=range(1000, 1000 + n_shifts))
    return np.clip(ct + rng.normal(0, noise_sd, ct.size), 0, None), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--shifts-per-trial", type=int, default=3)
    ap.add_argument("--mlflow", action="store_true", help="log trials and result to MLflow (./mlruns)")
    args = ap.parse_args()
    tracker = Tracker(args.mlflow, "warehouse-calibration")

    base = SimConfig()
    real_ct, real_thr = make_real_site_log(base)
    print(f"observed site: {real_ct.size} orders, mean cycle {real_ct.mean():.1f} min, {real_thr:.1f} orders/h")

    def objective(trial: optuna.Trial) -> float:
        cand = base.with_(
            pick_time_mean_min=trial.suggest_float("pick_time_mean_min", 0.2, 1.2),
            robot_speed_m_per_min=trial.suggest_float("robot_speed_m_per_min", 30.0, 90.0),
            congestion=trial.suggest_float("congestion", 0.0, 0.25),
        )
        # common random numbers across trials -> less noisy comparison between candidates
        sim_ct, sim_thr = simulate(cand, seeds=range(args.shifts_per_trial))
        gap = wasserstein_1d(sim_ct, real_ct) + abs(sim_thr - real_thr) / max(real_thr, 1e-9) * 10.0
        trial.set_user_attr("throughput", sim_thr)
        tracker.metrics({"gap": gap, "sim_throughput": sim_thr, **trial.params}, step=trial.number)
        return gap

    with tracker.run("optuna-tpe", trials=args.trials, shifts_per_trial=args.shifts_per_trial, sampler="TPE", seed=0,
                     observed_orders=int(real_ct.size)):
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        report = summarise(base, study, real_ct, args.shifts_per_trial)
        tracker.metrics({"gap_uncalibrated_default": report["gap_uncalibrated_default"],
                         "gap_calibrated_heldout_seeds": report["gap_calibrated_heldout_seeds"]})
        tracker.metrics(report["best_params"], prefix="best_")
        tracker.dict(report, "calibration_report.json")
    print(json.dumps(report, indent=2))


def summarise(base, study, real_ct, shifts):

    default_ct, _ = simulate(base, seeds=range(shifts))
    best_ct, _ = simulate(base.with_(**study.best_params), seeds=range(10, 10 + shifts))  # held-out seeds
    return {
        "gap_uncalibrated_default": round(wasserstein_1d(default_ct, real_ct), 3),
        "gap_calibrated_heldout_seeds": round(wasserstein_1d(best_ct, real_ct), 3),
        "best_params": {k: round(v, 3) for k, v in study.best_params.items()},
        "hidden_truth": HIDDEN_TRUTH,
        "note": "parameters can trade off (e.g. slower picking vs more congestion); check identifiability, not just the gap",
    }


if __name__ == "__main__":
    main()
