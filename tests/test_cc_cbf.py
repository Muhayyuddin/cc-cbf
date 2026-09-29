import math
import random

import numpy as np
import pytest

import algorithms.cc_cbf as ccbf
from algorithms.cc_cbf import CCCBFController
from core.entities import EncounterInfo, ObstacleState, VesselState
from core.scenario import get_scenario
from core.simulator import Simulator

ENCOUNTERS = list(ccbf.LAMBDA)


def test_default_parameters_match_paper_table_2():
    assert ccbf.LAMBDA == {"head_on": 0.55, "crossing_give_way": 0.60,
                           "crossing_stand_on": 0.10, "overtaking": 0.45, "none": 0.0}
    assert {k: round(math.degrees(v)) for k, v in ccbf.THETA_C.items()} == {
        "head_on": -45, "crossing_give_way": 60, "crossing_stand_on": 60,
        "overtaking": -30, "none": 0}
    assert (ccbf.ALPHA, ccbf.GAMMA, ccbf.ACTIVATION_RANGE) == (0.8, 4.0, 95.0)
    assert round(ccbf.R_BASE, 1) == 15.7


@pytest.mark.parametrize("enc", ENCOUNTERS)
def test_lobe_peaks_at_theta_c(enc):
    tc = ccbf.THETA_C[enc]
    r_peak = CCCBFController.r_cc_at_bearing(tc, enc)
    assert math.isclose(r_peak, ccbf.R_BASE * (1 + ccbf.LAMBDA[enc]))
    for th in np.linspace(-math.pi, math.pi, 73):
        assert CCCBFController.r_cc_at_bearing(th, enc) <= r_peak + 1e-12
        # opposite half-plane: isotropic radius
        if math.cos(th - tc) <= 0:
            assert math.isclose(CCCBFController.r_cc_at_bearing(th, enc), ccbf.R_BASE)


@pytest.mark.parametrize("enc", ["head_on", "crossing_give_way", "overtaking"])
def test_phi_derivative_is_consistent(enc):
    eps = 1e-6
    for th in np.linspace(-3.0, 3.0, 61):
        num = (ccbf._phi_and_dphi(th + eps, enc)[0] - ccbf._phi_and_dphi(th - eps, enc)[0]) / (2 * eps)
        assert math.isclose(ccbf._phi_and_dphi(th, enc)[1], num, abs_tol=1e-6)


def _random_case(rng):
    state = VesselState(x=rng.uniform(-40, 40), y=rng.uniform(-40, 40),
                        psi=rng.uniform(-math.pi, math.pi), u=rng.uniform(0, 4), r=0.0)
    obs = ObstacleState(x=rng.uniform(-80, 80), y=rng.uniform(-80, 80),
                        psi=rng.uniform(-math.pi, math.pi), speed=rng.uniform(0, 3),
                        length=rng.choice([8.0, 10.0]), width=rng.choice([3.0, 4.0]))
    return state, obs, rng.choice(ENCOUNTERS)


def test_scalar_row_matches_reference_constraint():
    rng = random.Random(1)
    for _ in range(2000):
        state, obs, enc = _random_case(rng)
        a, c, h, c_hard = CCCBFController._cbf_constraint(state, obs, EncounterInfo(enc))
        dx, dy = state.x - obs.x, state.y - obs.y
        hs, ax, ay, cs, chs = ccbf._row_scalar(state.x, state.y, state.psi, state.u, obs, enc,
                                               True, dx, dy, dx * dx + dy * dy)
        scale = 1.0 + abs(c)
        assert math.isclose(hs, h, rel_tol=1e-12, abs_tol=1e-9)
        assert math.isclose(ax, a[0], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(ay, a[1], rel_tol=1e-9, abs_tol=1e-9)
        assert abs(cs - c) <= 1e-9 * scale and abs(chs - c_hard) <= 1e-9 * scale
        # h_cc (used for plotting) is the same barrier
        h2 = CCCBFController.h_cc(state.x, state.y, state.psi, obs.x, obs.y, obs.psi,
                                  obs.length, obs.width, enc)
        assert math.isclose(h2, h, rel_tol=1e-9, abs_tol=1e-6)


def _random_qp(rng, n):
    u_nom = np.array([rng.uniform(-4, 4), rng.uniform(-4, 4)])
    A = np.array([[rng.uniform(-50, 50), rng.uniform(-50, 50)] for _ in range(n)])
    c = np.array([rng.uniform(-150, 100) for _ in range(n)])
    return u_nom, A, c


def test_scalar_qp_matches_array_qp():
    rng = random.Random(2)
    v_max = 4.1152
    for _ in range(3000):
        u_nom, A, c = _random_qp(rng, rng.randint(1, 4))
        ref = ccbf._qp2d(u_nom, A, c, v_max)
        fast = ccbf._qp2d_s(u_nom[0], u_nom[1], [tuple(a) + (ci,) for a, ci in zip(A, c)], v_max)
        assert (ref is None) == (fast is None)
        if ref is not None:
            assert np.allclose(ref, fast, atol=1e-9)


def test_exact_qp_is_optimal():
    """No feasible point on a fine polar grid beats the enumerated optimum."""
    rng = random.Random(3)
    v_max = 4.1152
    rr, tt = np.meshgrid(np.linspace(0, v_max, 120), np.linspace(-np.pi, np.pi, 720))
    grid = np.stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()], axis=1)
    checked = 0
    for _ in range(300):
        u_nom, A, c = _random_qp(rng, rng.randint(1, 3))
        u = ccbf._qp2d(u_nom, A, c, v_max)
        feasible = np.all(grid @ A.T - c >= -1e-9, axis=1)
        if u is None:
            assert not feasible.any()
            continue
        assert np.linalg.norm(u) <= v_max * (1 + 1e-9) + 1e-9
        assert np.all(A @ u - c >= -1e-7 * (1 + np.abs(c).max()))
        if feasible.any():
            best_grid = np.min(np.sum((grid[feasible] - u_nom) ** 2, axis=1))
            assert np.sum((u - u_nom) ** 2) <= best_grid + 1e-9
            checked += 1
    assert checked > 100


def test_constraint_priority_returns_command_when_infeasible():
    # Two opposing constraints that cannot both hold within the speed limit
    rows4 = [(1.0, 0.0, 10.0, 10.0), (-1.0, 0.0, 10.0, 10.0)]
    u = ccbf._solve_s(1.0, 0.0, rows4, 4.0)
    assert u is not None and math.hypot(*u) <= 4.0 + 1e-9


def test_overridden_restores_parameters():
    before = (dict(ccbf.LAMBDA), dict(ccbf.THETA_C), ccbf.GAMMA, ccbf.STERN_PASS_LAMBDA)
    with ccbf.overridden(gamma=0.0, theta_c={"head_on": 0.0}, **{"lambda": {"overtaking": 0.9}}):
        assert ccbf.GAMMA == 0.0 and ccbf.THETA_C["head_on"] == 0.0
        assert ccbf.LAMBDA["overtaking"] == 0.9
    assert (dict(ccbf.LAMBDA), dict(ccbf.THETA_C), ccbf.GAMMA, ccbf.STERN_PASS_LAMBDA) == before
    with pytest.raises(RuntimeError):
        with ccbf.overridden(stern_pass_lambda=0.0):
            raise RuntimeError
    assert ccbf.STERN_PASS_LAMBDA == before[3]
    with pytest.raises(KeyError):
        with ccbf.overridden(no_such_parameter=1):
            pass


@pytest.mark.parametrize("scenario", ["head_on", "crossing_give_way", "overtaking"])
def test_fast_and_reference_paths_agree(scenario):
    def run(fast):
        with ccbf.overridden(fast_step=fast):
            sim = Simulator(get_scenario(scenario, seed=42), CCCBFController())
            sim.run_to_completion()
        return [(r.state.x, r.state.y) for r in sim.records]
    a, b = run(True), run(False)
    assert len(a) == len(b)
    assert np.allclose(a, b, atol=1e-6)
