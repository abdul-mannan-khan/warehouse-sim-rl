"""Render a side-by-side demo video: the same shift of orders under two dispatch policies.

    python render_video.py --minutes 240 --seconds 60 --out media/demo.mp4
    python render_video.py --model ppo_dispatch.zip      # third panel with a trained PPO policy
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from warehouse_sim import SimConfig, WarehouseSim, WarehouseDispatchEnv
from warehouse_sim.policies import BASELINES

try:  # use the ffmpeg bundled with imageio-ffmpeg when no system ffmpeg is installed
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

BLUE, RED, GREEN, GREY = "#1f5fa8", "#d1402f", "#2e9e5b", "#9aa3ad"


def run_policy(cfg, policy_fn):
    sim = WarehouseSim(cfg, policy=policy_fn)
    sim.run()
    return sim


def run_agent(cfg, model):
    env = WarehouseDispatchEnv(cfg, randomise_seed=False)
    obs, _ = env.reset()
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            return env.sim


class Panel:
    def __init__(self, ax, sim: WarehouseSim, title: str):
        self.ax, self.sim, self.cfg = ax, sim, sim.cfg
        self.orders = sim.orders
        d = np.asarray(self.cfg.zone_distance_m)
        self.zone_xy = np.stack([d, np.arange(self.cfg.n_zones) * 1.0], axis=1)
        self.depot = np.array([0.0, (self.cfg.n_zones - 1) / 2])
        ax.set_xlim(-14, d.max() + 34); ax.set_ylim(-0.9, self.cfg.n_zones - 0.2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        for z, (x, y) in enumerate(self.zone_xy):
            ax.plot([0, x], [self.depot[1], y], color="#e3e7eb", lw=1, zorder=0)
            ax.add_patch(plt.Rectangle((x - 3, y - 0.32), 6, 0.64, color="#eef2f6", zorder=0))
            ax.text(x, y + 0.42, f"Zone {z + 1}", ha="center", fontsize=8, color="#555")
        ax.add_patch(plt.Rectangle((-6, self.depot[1] - 0.45), 6, 0.9, color="#dfe6ee", zorder=0))
        ax.text(-3, self.depot[1] - 0.62, "depot +\npacking", ha="center", va="top", fontsize=8, color="#555")
        self.robots = ax.scatter([], [], s=70, marker="s", zorder=3, edgecolors="white", linewidths=0.8)
        self.backlog_txt = [ax.text(x + 4.5, y, "", va="center", fontsize=9) for x, y in self.zone_xy]
        self.pack_txt = ax.text(-3, self.depot[1] + 0.6, "", ha="center", fontsize=8)
        self.kpi = ax.text(-12, -0.8, "", fontsize=10, family="monospace", va="bottom")

    def update(self, t: float):
        pts, cols = [], []
        waiting = np.zeros(self.cfg.n_zones, int)
        wait_rush = np.zeros(self.cfg.n_zones, int)
        packing = 0
        stack = np.zeros(self.cfg.n_zones)
        for o in self.orders:
            if o.created > t:
                break
            if o.dispatched is None or o.dispatched > t:
                waiting[o.zone] += 1; wait_rush[o.zone] += o.rush
                continue
            zxy = self.zone_xy[o.zone]
            col = RED if o.rush else BLUE
            if o.at_zone is not None and t < o.at_zone:            # driving out
                f = (t - o.dispatched) / max(o.at_zone - o.dispatched, 1e-9)
                p = self.depot + f * (zxy - self.depot)
            elif o.pick_end is not None and t < o.pick_end:          # queueing for / being served by a picker
                p = zxy + np.array([-1.5 + 1.6 * (stack[o.zone] % 3), 0.18 - 0.2 * (stack[o.zone] // 3)])
                stack[o.zone] += 1
                col = GREEN if (o.pick_start is not None and t >= o.pick_start) else col
            elif o.back is not None and t < o.back:                  # driving back
                f = (t - o.pick_end) / max(o.back - o.pick_end, 1e-9)
                p = zxy + f * (self.depot - zxy)
            else:
                if o.done is None or t < o.done:
                    packing += 1
                continue
            pts.append(p); cols.append(col)
        self.robots.set_offsets(np.array(pts) if pts else np.empty((0, 2)))
        self.robots.set_facecolors(cols if cols else [])
        for z, txt in enumerate(self.backlog_txt):
            txt.set_text(("■" * min(waiting[z], 12)) + (f" {waiting[z]}" if waiting[z] else ""))
            txt.set_color(RED if wait_rush[z] else "#333")
        self.pack_txt.set_text(f"packing queue {packing}" if packing else "")
        done = [o for o in self.orders if o.done is not None and o.done <= t]
        late = sum(o.late() for o in done)
        ct = np.mean([o.done - o.created for o in done]) if done else 0.0
        backlog = int(waiting.sum())
        self.kpi.set_text(f"done {len(done):4d}   late {late:3d}   mean cycle {ct:5.1f} min   waiting {backlog:3d}")
        return backlog, late


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=480.0, help="simulated minutes shown")
    ap.add_argument("--seconds", type=float, default=60.0, help="video length")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=None, help="optional PPO .zip for a third panel")
    ap.add_argument("--out", default="media/demo.mp4")
    args = ap.parse_args()

    cfg = SimConfig(seed=args.seed, horizon_min=args.minutes)
    sims = [("First-in-first-out dispatch", run_policy(cfg, BASELINES["fifo"])),
            ("Least-loaded-zone dispatch", run_policy(cfg, BASELINES["least_loaded_zone"]))]
    if args.model:
        from stable_baselines3 import PPO
        sims.append(("PPO dispatch (learned)", run_agent(cfg, PPO.load(args.model))))

    n = len(sims)
    fig = plt.figure(figsize=(6.4 * n, 7.2), dpi=100)
    gs = fig.add_gridspec(2, n, height_ratios=[3.2, 1.25], hspace=0.28, wspace=0.05,
                          left=0.055, right=0.98, top=0.86, bottom=0.08)
    fig.suptitle("Same orders, different dispatch policies: SimPy warehouse simulator", fontsize=16, fontweight="bold", x=0.03, ha="left")
    fig.text(0.03, 0.905, f"{cfg.n_robots} robots, {cfg.n_zones} picking zones, {cfg.arrival_rate_per_min * 60:.0f} orders/hour; "
             "blue = normal order, red = rush order, green = being picked", fontsize=10, color="#444")
    clock = fig.text(0.98, 0.905, "", ha="right", fontsize=12, family="monospace")
    panels = [Panel(fig.add_subplot(gs[0, i]), sim, title) for i, (title, sim) in enumerate(sims)]

    axb = fig.add_subplot(gs[1, :])
    axb.set_xlim(0, args.minutes); axb.set_xlabel("simulated minutes"); axb.set_ylabel("orders waiting")
    axb.spines[["top", "right"]].set_visible(False)
    colours = ["#7a8691", BLUE, GREEN]
    lines = [axb.plot([], [], lw=2, color=colours[i], label=t)[0] for i, (t, _) in enumerate(sims)]
    axb.legend(loc="upper left", frameon=False)
    hist_t, hist = [], [[] for _ in sims]

    frames = int(args.seconds * args.fps)
    times = np.linspace(0, args.minutes, frames)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=2500, metadata={"title": "warehouse-sim-rl demo"})
    with writer.saving(fig, args.out, dpi=100):
        for t in times:
            hist_t.append(t)
            for i, p in enumerate(panels):
                backlog, _ = p.update(t)
                hist[i].append(backlog)
                lines[i].set_data(hist_t, hist[i])
            axb.set_ylim(0, max(10, max(max(h) for h in hist) * 1.15))
            clock.set_text(f"t = {int(t) // 60:02d}:{int(t) % 60:02d}")
            writer.grab_frame()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
