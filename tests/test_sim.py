import numpy as np
from gymnasium.utils.env_checker import check_env

from warehouse_sim import SimConfig, WarehouseSim, WarehouseDispatchEnv
from warehouse_sim.policies import BASELINES

CFG = SimConfig(horizon_min=120.0)


def test_same_seed_same_result():
    a = WarehouseSim(CFG, policy=BASELINES["fifo"]).run()
    b = WarehouseSim(CFG, policy=BASELINES["fifo"]).run()
    assert a == b


def test_orders_are_conserved():
    sim = WarehouseSim(CFG, policy=BASELINES["rush_first"])
    sim.run()
    in_flight = len(sim.orders) - len(sim.completed) - len(sim.backlog.items)
    assert in_flight >= 0
    assert in_flight <= CFG.n_robots + CFG.n_packers + sum(CFG.pickers_per_zone) + 50  # bounded work in progress


def test_resources_never_over_capacity():
    sim = WarehouseSim(CFG, policy=BASELINES["least_loaded_zone"])
    while sim.env.peek() < CFG.horizon_min:
        sim.env.step()
        assert sim.robots.count <= CFG.n_robots
        assert all(p.count <= c for p, c in zip(sim.pickers, CFG.pickers_per_zone))


def test_rush_orders_jump_the_packing_queue():
    cfg = CFG.with_(n_packers=1, pack_time_mean_min=2.0, rush_prob=0.3)   # make packing the bottleneck
    sim = WarehouseSim(cfg, policy=BASELINES["fifo"])
    sim.run()
    rush = [o.done - o.dispatched for o in sim.completed if o.rush]
    normal = [o.done - o.dispatched for o in sim.completed if not o.rush]
    assert np.mean(rush) < np.mean(normal)


def test_gymnasium_api():
    check_env(WarehouseDispatchEnv(CFG), skip_render_check=True)


def test_episode_runs_to_end():
    env = WarehouseDispatchEnv(CFG)
    env.reset(seed=1)
    steps, done = 0, False
    while not done:
        _, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        done, steps = term or trunc, steps + 1
    assert steps > 10 and "throughput_per_hour" in info
