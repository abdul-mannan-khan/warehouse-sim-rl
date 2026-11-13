"""Heuristic dispatch policies: the baselines any learned policy must beat."""
from __future__ import annotations

import numpy as np


def fifo(sim) -> int:
    """Serve the oldest waiting order."""
    return sim.backlog.items[0].zone


def rush_first(sim) -> int:
    """Oldest rush order if any, otherwise oldest order."""
    for o in sim.backlog.items:
        if o.rush:
            return o.zone
    return sim.backlog.items[0].zone


def least_loaded_zone(sim) -> int:
    """Among zones with waiting work, pick the one whose pickers are least busy (ties: oldest work)."""
    s = sim.zone_summary()
    candidates = np.flatnonzero(s[:, 0] > 0)
    # sort by picker load, then by -age (older first)
    return int(min(candidates, key=lambda z: (s[z, 3], -s[z, 2])))


BASELINES = {"fifo": fifo, "rush_first": rush_first, "least_loaded_zone": least_loaded_zone}
