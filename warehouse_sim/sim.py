"""Discrete-event simulation of a small robot-assisted picking warehouse, built on SimPy.

Flow of one order
-----------------
1. Orders arrive (Poisson process) and wait in a backlog       -> simpy.FilterStore (a Store)
2. A free robot is assigned an order by a dispatch policy       -> robots: simpy.Resource
3. The robot drives to the order's zone (slower when many
   robots are moving: a simple congestion model)
4. It waits for a human picker in that zone, who picks the lines -> pickers: one simpy.Resource per zone
5. The robot returns to the depot and is freed
6. The tote queues for the packing station; rush orders go first -> packing: simpy.PriorityResource

Time unit: minutes. Distance unit: metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import simpy


@dataclass(frozen=True)
class SimConfig:
    n_robots: int = 6
    pickers_per_zone: tuple[int, ...] = (1, 1, 1, 1)
    zone_distance_m: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)  # depot -> zone
    zone_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)          # where orders come from
    arrival_rate_per_min: float = 1.8
    rush_prob: float = 0.15
    max_lines: int = 4                    # lines (picks) per order: uniform 1..max_lines
    robot_speed_m_per_min: float = 60.0   # 1 m/s
    congestion: float = 0.05              # +5% travel time per other robot on the move
    pick_time_mean_min: float = 0.5       # per line, exponential
    pack_time_mean_min: float = 0.45      # per tote, exponential
    n_packers: int = 1
    sla_rush_min: float = 20.0
    sla_normal_min: float = 60.0
    horizon_min: float = 480.0            # one 8-hour shift
    seed: int = 0

    @property
    def n_zones(self) -> int:
        return len(self.zone_distance_m)

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)


@dataclass
class Order:
    id: int
    created: float
    zone: int
    lines: int
    rush: bool
    dispatched: Optional[float] = None
    at_zone: Optional[float] = None       # robot reached the zone
    pick_start: Optional[float] = None
    pick_end: Optional[float] = None
    back: Optional[float] = None          # robot back at the depot
    done: Optional[float] = None

    @property
    def sla(self) -> float:  # filled in by the simulator
        return self._sla

    def late(self) -> bool:
        return self.done is not None and (self.done - self.created) > self._sla


# A policy looks at the simulator and returns the zone to serve next.
Policy = Callable[["WarehouseSim"], int]


class WarehouseSim:
    """Run with a fixed `policy`, or leave policy=None and drive decisions externally
    (that is what the Gymnasium wrapper in env.py does)."""

    def __init__(self, cfg: SimConfig, policy: Optional[Policy] = None):
        assert len(cfg.pickers_per_zone) == cfg.n_zones == len(cfg.zone_weights)
        self.cfg = cfg
        self.policy = policy
        # separate random streams: demand does not change when the policy or service times change
        # (common random numbers -> fair comparison between policies and calibration candidates)
        ss = np.random.SeedSequence(cfg.seed)
        self.rng_demand, self.rng_service = (np.random.default_rng(x) for x in ss.spawn(2))
        self.env = simpy.Environment()

        self.backlog = simpy.FilterStore(self.env)
        self.robots = simpy.Resource(self.env, capacity=cfg.n_robots)
        self.pickers = [simpy.Resource(self.env, capacity=c) for c in cfg.pickers_per_zone]
        self.packing = simpy.PriorityResource(self.env, capacity=cfg.n_packers)

        self.orders: list[Order] = []
        self.completed: list[Order] = []
        self.robots_travelling = 0

        # external-decision plumbing (used when policy is None)
        self.awaiting_decision = False
        self._action_evt: Optional[simpy.Event] = None
        self._arrival_evt: Optional[simpy.Event] = None
        self.reward_buffer = 0.0      # rewards earned since the last decision
        self.invalid_actions = 0

        self.env.process(self._order_source())
        self.env.process(self._dispatcher())

    # ------------------------------------------------------------------ processes
    def _order_source(self):
        cfg, oid = self.cfg, 0
        weights = np.asarray(cfg.zone_weights) / np.sum(cfg.zone_weights)
        while True:
            yield self.env.timeout(self.rng_demand.exponential(1.0 / cfg.arrival_rate_per_min))
            rush = bool(self.rng_demand.random() < cfg.rush_prob)
            o = Order(oid, self.env.now, int(self.rng_demand.choice(cfg.n_zones, p=weights)),
                      int(self.rng_demand.integers(1, cfg.max_lines + 1)), rush)
            o._sla = cfg.sla_rush_min if rush else cfg.sla_normal_min
            oid += 1
            self.orders.append(o)
            self.backlog.put(o)
            if self._arrival_evt is not None and not self._arrival_evt.triggered:
                self._arrival_evt.succeed()

    def _dispatcher(self):
        while True:
            robot = self.robots.request()
            yield robot                                   # wait for a free robot
            while not self.backlog.items:                 # wait for work
                self._arrival_evt = self.env.event()
                yield self._arrival_evt
            zone = yield from self._decide()
            if not any(o.zone == zone for o in self.backlog.items):
                self.invalid_actions += 1                 # empty zone chosen: fall back to oldest order
                zone = self.backlog.items[0].zone
            in_zone = [o for o in self.backlog.items if o.zone == zone]     # oldest first
            target = next((o for o in in_zone if o.rush), in_zone[0])      # rush orders first within the zone
            order = yield self.backlog.get(lambda o, t=target: o is t)
            order.dispatched = self.env.now
            self.env.process(self._fulfil(order, robot))

    def _decide(self):
        if self.policy is not None:
            return int(self.policy(self))
        self.awaiting_decision = True
        self._action_evt = self.env.event()
        action = yield self._action_evt
        self.awaiting_decision = False
        return int(action)

    def _fulfil(self, order: Order, robot):
        cfg = self.cfg
        travel = cfg.zone_distance_m[order.zone] / cfg.robot_speed_m_per_min
        travel *= 1.0 + cfg.congestion * self.robots_travelling
        self.robots_travelling += 1
        yield self.env.timeout(travel)
        self.robots_travelling -= 1
        order.at_zone = self.env.now

        with self.pickers[order.zone].request() as picker:
            yield picker
            order.pick_start = self.env.now
            yield self.env.timeout(sum(self.rng_service.exponential(cfg.pick_time_mean_min) for _ in range(order.lines)))
            order.pick_end = self.env.now

        self.robots_travelling += 1
        yield self.env.timeout(travel)
        self.robots_travelling -= 1
        order.back = self.env.now
        self.robots.release(robot)

        with self.packing.request(priority=0 if order.rush else 1) as packer:
            yield packer
            yield self.env.timeout(self.rng_service.exponential(cfg.pack_time_mean_min))

        order.done = self.env.now
        self.completed.append(order)
        self.reward_buffer += (-2.0 if order.rush else -1.0) if order.late() else 0.1

    # ------------------------------------------------------------------ driving
    def run(self) -> dict:
        """Run a whole shift with the built-in policy."""
        assert self.policy is not None, "run() needs a policy; use the Gymnasium env otherwise"
        self.env.run(until=self.cfg.horizon_min)
        return self.metrics()

    def advance_to_next_decision(self) -> bool:
        """Step the event loop until a decision is needed. Returns False when the shift is over."""
        while not self.awaiting_decision:
            if self.env.peek() >= self.cfg.horizon_min:
                return False
            self.env.step()
        return True

    def act(self, zone: int) -> None:
        self._action_evt.succeed(zone)
        self.awaiting_decision = False   # resumes on the next env.step()

    # ------------------------------------------------------------------ observation helpers
    def zone_summary(self) -> np.ndarray:
        """Per zone: waiting orders, rush waiting, age of oldest waiting order (min), picker queue length."""
        out = np.zeros((self.cfg.n_zones, 4), dtype=np.float32)
        now = self.env.now
        for o in self.backlog.items:
            out[o.zone, 0] += 1
            out[o.zone, 1] += o.rush
            out[o.zone, 2] = max(out[o.zone, 2], now - o.created)
        for z, p in enumerate(self.pickers):
            out[z, 3] = len(p.queue) + p.count
        return out

    def metrics(self) -> dict:
        done = self.completed
        ct = np.array([o.done - o.created for o in done]) if done else np.array([np.nan])
        return {
            "generated": len(self.orders),
            "completed": len(done),
            "throughput_per_hour": len(done) / (self.cfg.horizon_min / 60.0),
            "cycle_time_mean_min": float(np.nanmean(ct)),
            "cycle_time_p95_min": float(np.nanpercentile(ct, 95)),
            "late_rate": float(np.mean([o.late() for o in done])) if done else float("nan"),
            "rush_late_rate": float(np.mean([o.late() for o in done if o.rush] or [np.nan])),
            "backlog_end": len(self.backlog.items),
            "invalid_actions": self.invalid_actions,
        }

    def cycle_times(self) -> np.ndarray:
        return np.array([o.done - o.created for o in self.completed])
