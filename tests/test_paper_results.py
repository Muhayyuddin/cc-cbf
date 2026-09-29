"""Regression tests against the deterministic results of paper Table III (seed 42)."""
import pytest

from algorithms.registry import CONTROLLERS
from core.scenario import get_scenario
from core.simulator import Simulator

# (controller, scenario): (min separation [m], COL, collision)  -- Table III
TABLE3 = {
    ("CC-CBF", "head_on"): (19.15, 1.00, False),
    ("CC-CBF", "crossing_give_way"): (28.92, 1.00, False),
    ("CC-CBF", "overtaking"): (15.96, 1.00, False),
    ("CC-CBF", "static_obstacles"): (12.74, 1.00, False),
    ("C3BF", "head_on"): (27.00, 1.00, False),
    ("C3BF", "crossing_give_way"): (28.28, 0.20, False),
    ("C3BF", "overtaking"): (16.22, 0.00, False),
    ("C3BF", "static_obstacles"): (13.77, 1.00, False),
    ("Rule-COLREG", "head_on"): (4.61, 1.00, False),
    ("Rule-COLREG", "crossing_give_way"): (8.57, 1.00, False),
    ("Rule-COLREG", "overtaking"): (9.03, 1.00, False),
    ("Rule-COLREG", "static_obstacles"): (9.21, 1.00, False),
    ("Geo-CRI", "head_on"): (0.00, 0.91, True),
    ("Geo-CRI", "crossing_give_way"): (9.49, 0.72, False),
    ("Geo-CRI", "overtaking"): (8.03, 0.52, False),
    ("Geo-CRI", "static_obstacles"): (9.06, 1.00, False),
    ("TC-CBF", "head_on"): (12.64, 1.00, False),
    ("TC-CBF", "crossing_give_way"): (15.24, 0.67, False),
    ("TC-CBF", "overtaking"): (10.73, 1.00, False),
    ("TC-CBF", "static_obstacles"): (12.74, 1.00, False),
}
# Runs that take more than a few seconds (predictive baselines)
SLOW = {("Rule-COLREG", "static_obstacles"), ("Geo-CRI", "overtaking"),
        ("Geo-CRI", "static_obstacles")}


def _case(key):
    return pytest.param(key, marks=pytest.mark.slow) if key in SLOW else key


@pytest.mark.parametrize("key", [_case(k) for k in TABLE3], ids=lambda k: f"{k[0]}-{k[1]}")
def test_table3(key):
    name, scenario = key
    sep, col, coll = TABLE3[key]
    sim = Simulator(get_scenario(scenario, seed=42), CONTROLLERS[name]())
    sim.run_to_completion()
    m = sim.get_metrics()
    assert round(m.min_separation, 2) == sep
    assert round(m.colreg_compliance, 2) == col
    assert m.collision_flag == coll


def test_cc_cbf_keeps_clearance_above_d_safe_in_mc_sample():
    """CC-CBF stays collision-free and above D_safe on a Monte Carlo sample
    (a few static-obstacle trials do not reach the goal disc within 180 s)."""
    import run_mc_5way as mc
    for scenario in mc.SCENARIOS:
        for seed in range(3):
            r = mc.run_single(CONTROLLERS["CC-CBF"], scenario, seed)
            assert r["collision"] == 0 and r["min_separation"] > 8.0
            if scenario != "static_obstacles":
                assert r["reached_goal"] == 1
