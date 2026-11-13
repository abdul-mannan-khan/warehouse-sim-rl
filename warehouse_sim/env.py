"""Gymnasium wrapper: each step = one dispatch decision (which zone a newly free robot should serve)."""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sim import SimConfig, WarehouseSim


class WarehouseDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig = SimConfig(), randomise_seed: bool = True):
        super().__init__()
        self.base_cfg = cfg
        self.randomise_seed = randomise_seed
        z = cfg.n_zones
        # per zone: waiting/10, rush waiting/5, oldest age/SLA, picker load/5 ; then time-of-shift, robots free
        self.observation_space = spaces.Box(0.0, 1e3, shape=(z * 4 + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(z)
        self.sim: Optional[WarehouseSim] = None

    def _obs(self) -> np.ndarray:
        s = self.sim.zone_summary()
        s = s / np.array([10.0, 5.0, self.base_cfg.sla_normal_min, 5.0], dtype=np.float32)
        extra = np.array([self.sim.env.now / self.base_cfg.horizon_min,
                          (self.sim.robots.capacity - self.sim.robots.count) / self.sim.robots.capacity],
                         dtype=np.float32)
        return np.concatenate([s.ravel(), extra]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Zones that currently have waiting orders (usable with MaskablePPO from sb3-contrib)."""
        return self.sim.zone_summary()[:, 0] > 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        run_seed = int(self.np_random.integers(1 << 31)) if self.randomise_seed else self.base_cfg.seed
        self.sim = WarehouseSim(self.base_cfg.with_(seed=run_seed), policy=None)
        self.sim.advance_to_next_decision()
        return self._obs(), {}

    def step(self, action):
        invalid_before = self.sim.invalid_actions
        self.sim.act(int(action))
        alive = self.sim.advance_to_next_decision()
        reward = self.sim.reward_buffer - 0.2 * (self.sim.invalid_actions - invalid_before)
        self.sim.reward_buffer = 0.0
        terminated = False
        truncated = not alive          # end of shift is a time limit, not a terminal state
        info = self.sim.metrics() if truncated else {}
        return self._obs(), float(reward), terminated, truncated, info
