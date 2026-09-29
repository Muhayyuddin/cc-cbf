"""
CC-CBF -- Compliant-Course Control Barrier Function
===================================================
A control barrier function for maritime collision avoidance that encodes
collision safety and the COLREG passing direction in a single barrier per
target (paper Sec. IV).

Barrier
-------
    h_CC(x) = ||p_o - p_t||^2 - R_CC(theta_rel, tau)^2

    R_CC(theta, tau) = R_b * [1 + lambda(tau) * Phi(theta, tau)]
    Phi(theta, tau)  = [cos(theta - theta_C(tau))]_+^2          (C^1 lobe)

    lambda(tau)  -- encounter-dependent asymmetry strength   (Table II)
    theta_C(tau) -- lobe centre in the body frame            (Table II)

For crossing give-way the stern-passage factor
    S = 1 + STERN_PASS_LAMBDA * [cos(phi_world - psi_t)]_+
inflates the barrier ahead of the target's bow (Rule 15, "do not cross ahead").

Constraint and QP
-----------------
With the barrier orientation frozen at the measured heading during one
control update, the CBF condition  dh/dt >= -alpha * h  is affine in the
decision variable u = v_o (world-frame velocity):

    a^T u >= -alpha * h - b + gamma * (v_close * d + proximity * R_CC)

where the last term is the anticipatory tightening.  The planar QP

    min ||u - u_nom||^2   s.t.  A u >= c,  ||u|| <= v_max

is solved exactly by enumerating the O(n^2) candidate active sets
(Algorithm 1).  If the tightened problem is infeasible, the tightening is
relaxed by the smallest uniform factor; if even the barrier condition is
infeasible, the least-violating command is returned.

The nominal velocity u_nom is the cruise velocity towards the goal, biased
25 deg to starboard for head-on / overtaking (Layer 2) or steered towards a
point astern of the target at reduced speed for crossing give-way (Layer 3).

Parameters are module-level constants matching the paper; use
:func:`overridden` to change them temporarily (ablation / sensitivity).
Mean computation per control update is ~24 us on one CPU core.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import data.config as cfg
from algorithms.base_controller import BaseController, manoeuvre_label
from core.colregs import classify_encounter, classify_type_fast
from core.entities import (
    ControlCommand, EncounterInfo, ObstacleState, ScenarioConfig, VesselState,
)
from core.geometry import wrap_angle

# =====================================================================
# CC-CBF parameters  (paper Tables I-II and Sec. VI-A)
# =====================================================================

_OWN_HD = math.hypot(cfg.USV_LENGTH, cfg.USV_WIDTH) / 2.0
_OBS_HD = math.hypot(cfg.DEFAULT_TARGET_LENGTH,
                     cfg.DEFAULT_TARGET_WIDTH) / 2.0
# Base radius R_b for the default 8 x 3 m target (15.7 m); each target uses
# its own R_b from its dimensions (_obs_rb).
R_BASE = _OWN_HD + _OBS_HD + cfg.SAFETY_BUFFER

# Asymmetry strength lambda per encounter type
LAMBDA: Dict[str, float] = {
    "head_on":           0.55,
    "crossing_give_way": 0.60,
    "crossing_stand_on": 0.10,
    "overtaking":        0.45,
    "none":              0.00,
}

# Lobe centre theta_C (body frame, counter-clockwise; negative = starboard).
# By Lemma 1 the barrier correction moves the target bearing AWAY from theta_C.
THETA_C: Dict[str, float] = {
    "head_on":           math.radians(-45),  # inflate stbd -> starboard turn (Rule 14)
    "crossing_give_way": math.radians(60),   # inflate port -> pass astern (Rule 15)
    "crossing_stand_on": math.radians(60),   # inflate port -> minor (Rule 17)
    "overtaking":        math.radians(-30),  # pass the overtaken vessel on its starboard side (Rule 13)
    "none":              0.0,
}

# Class-K gain alpha
ALPHA = 0.8

# Anticipatory tightening gain gamma:
#   a^T u >= -alpha * h - b + gamma * (v_close * d + proximity * R_CC)
GAMMA = 4.0

# Rule 15 stern-passage factor (crossing give-way only):
#   R_eff = R_CC * [1 + STERN_PASS_LAMBDA * max(0, cos(phi_world - psi_t))]
# with phi_world the world bearing from the target to the own ship; the
# barrier is inflated when the own ship is ahead of the target's bow.
STERN_PASS_LAMBDA = 0.55

# Activation distance d_act: only targets closer than this enter the QP
ACTIVATION_RANGE = 95.0  # m

# Constraint form: "exact" = paper Eq. (18) (default); "legacy" = radial
# normal with the R_CC rate evaluated at the current velocity and yaw rate
# (the earlier controller used for the Gazebo runs and sea trials).
CONSTRAINT_MODE = "exact"
# Add the measured heading-rotation term K * r to the drift b.
YAW_RATE_COMP = False
# Keep a give-way class until the target is past CPA (encounter memory).
PERSIST_GIVE_WAY = True
_GIVE_WAY = ("head_on", "crossing_give_way", "overtaking")

# Nominal shaping (heading bias / stern goal) acts only until the give-way
# target is "past and clear" (Rule 8(d)): range opening AND beyond the
# avoidance domain; afterwards the nominal is the goal.
NOMINAL_CPA_GATE = True
# Heading-bias profile: "ramp" = delta * max(0, 1 - d/d_act); "full" = delta
# throughout d <= d_act (used in the paper).
BIAS_PROFILE = "full"
BIAS_DEG: Dict[str, float] = {"head_on": 25.0, "overtaking": 25.0}

# Layer 3 (crossing give-way): nominal steered to a point STERN_OFFSET m
# astern of the target at STERN_SPEED_FACTOR x cruise speed.
STERN_OFFSET = 35.0        # m
STERN_SPEED_FACTOR = 0.65

# Use the scalar implementation of the control step (same mathematics as
# the NumPy reference path, ~10x faster).
FAST_STEP = True

_TUNABLE = ("LAMBDA", "THETA_C", "ALPHA", "GAMMA", "STERN_PASS_LAMBDA",
            "ACTIVATION_RANGE", "CONSTRAINT_MODE", "YAW_RATE_COMP",
            "PERSIST_GIVE_WAY", "NOMINAL_CPA_GATE", "BIAS_PROFILE", "BIAS_DEG",
            "STERN_OFFSET", "STERN_SPEED_FACTOR", "FAST_STEP")


@contextmanager
def overridden(**params) -> Iterator[None]:
    """
    Temporarily override CC-CBF module parameters.

    Keyword names are the parameter names in lower or upper case, e.g.
    ``overridden(gamma=0.0, theta_c={"head_on": 0.0})``.  Dictionary
    parameters (LAMBDA, THETA_C, BIAS_DEG) are updated key-wise and in
    place.  The override is process-wide and not thread-safe; run parallel
    studies in separate processes.
    """
    g = globals()
    saved = {}
    try:
        for name, value in params.items():
            key = name.upper()
            if key not in _TUNABLE:
                raise KeyError(f"unknown CC-CBF parameter: {name}")
            current = g[key]
            if isinstance(current, dict):
                saved[key] = dict(current)
                current.update(value)
            else:
                saved[key] = current
                g[key] = value
        yield
    finally:
        for key, value in saved.items():
            if isinstance(g[key], dict):
                g[key].clear()
                g[key].update(value)
            else:
                g[key] = value


# =====================================================================
# Barrier helpers
# =====================================================================

def _obs_rb(length: float, width: float) -> float:
    """Base radius R_b: own and target half-diagonals plus the safety buffer."""
    return _OWN_HD + math.hypot(length, width) / 2.0 + cfg.SAFETY_BUFFER


_RB_CACHE: Dict[Tuple[float, float], float] = {}


def _rb_cached(length: float, width: float) -> float:
    key = (length, width)
    v = _RB_CACHE.get(key)
    if v is None:
        v = _obs_rb(length, width)
        _RB_CACHE[key] = v
    return v


def _phi_and_dphi(theta_rel: float, enc_type: str) -> Tuple[float, float]:
    """
    Squared rectified-cosine lobe (C^1 everywhere) and its derivative.

        Phi  = [cos(theta_rel - theta_C)]_+^2
        dPhi = -2 [cos(theta_rel - theta_C)]_+ sin(theta_rel - theta_C)
    """
    tc = THETA_C.get(enc_type, 0.0)
    diff = theta_rel - tc
    c = math.cos(diff)
    cp = max(0.0, c)
    phi = cp * cp
    dphi = -2.0 * cp * math.sin(diff) if cp > 0.0 else 0.0
    return phi, dphi


def _closing(state, obs) -> bool:
    """True while the range to *obs* is decreasing (CPA not yet passed)."""
    dx, dy = state.x - obs.x, state.y - obs.y
    dvx = state.u * math.cos(state.psi) - obs.speed * math.cos(obs.psi)
    dvy = state.u * math.sin(state.psi) - obs.speed * math.sin(obs.psi)
    return dx * dvx + dy * dvy < 0.0


def _not_clear(state, obs) -> bool:
    """True until *obs* is past and clear: range closing, or within the
    avoidance domain."""
    return (_closing(state, obs) or
            math.hypot(state.x - obs.x, state.y - obs.y) - _rb_cached(obs.length, obs.width)
            + cfg.SAFETY_BUFFER < cfg.AVOIDANCE_DOMAIN)


# =====================================================================
# Exact planar QP  (Algorithm 1) -- NumPy reference implementation
# =====================================================================

def _qp2d(u_nom, A, c, v_max, tol=1e-9):
    """Exact 2-D QP  min||u-u_nom||^2  s.t. A u >= c, ||u|| <= v_max.

    The optimum is the unconstrained point, a projection onto one active
    half-plane or onto the speed circle, or an intersection of two active
    boundaries; every candidate is checked for feasibility.
    Returns None when the feasible set is empty."""
    n = A.shape[0]
    # Fast path: the unconstrained minimizer u_nom is optimal whenever it is
    # feasible (same feasibility test as applied to every candidate below).
    nu = np.linalg.norm(u_nom)
    if nu <= v_max * (1 + 1e-9) + tol and (
            n == 0 or np.min(A @ u_nom - c) >= -tol * (1 + np.max(np.abs(c)))):
        return u_nom.copy()
    cands = [u_nom]
    cands.append(u_nom * (v_max / nu) if nu > 1e-12 else np.zeros(2))
    for i in range(n):
        ai, ci = A[i], c[i]
        aa = ai @ ai
        if aa < 1e-18:
            continue
        cands.append(u_nom + ((ci - ai @ u_nom) / aa) * ai)
        # line-circle intersections
        p0 = (ci / aa) * ai
        rem = v_max * v_max - p0 @ p0
        if rem >= 0:
            t = np.array([-ai[1], ai[0]]) / math.sqrt(aa)
            r = math.sqrt(rem)
            cands += [p0 + r * t, p0 - r * t]
        for j in range(i + 1, n):
            M = np.array([A[i], A[j]])
            if abs(np.linalg.det(M)) > 1e-12:
                cands.append(np.linalg.solve(M, np.array([c[i], c[j]])))
    best, bcost = None, float("inf")
    for u in cands:
        if np.linalg.norm(u) > v_max * (1 + 1e-9) + tol:
            continue
        if n and np.min(A @ u - c) < -tol * (1 + np.max(np.abs(c))):
            continue
        cost = float((u - u_nom) @ (u - u_nom))
        if cost < bcost:
            best, bcost = u.copy(), cost
    return best


# =====================================================================
# Scalar implementation of one control step (planar problem, few targets):
# identical mathematics to CCCBFController._cbf_constraint / _solve_cbf_qp,
# written with float arithmetic to avoid array overhead (FAST_STEP).
# =====================================================================

def _row_scalar(sx, sy, spsi, su, obs, enc_type, need_row, dpx, dpy, d2):
    """Return (h, ax, ay, c, c_hard) for one obstacle (row entries None if
    not needed).  Same formulas as CCCBFController._cbf_constraint."""
    cos, sin, atan2, sqrt = math.cos, math.sin, math.atan2, math.sqrt
    rb = _rb_cached(obs.length, obs.width)
    lam = LAMBDA.get(enc_type, 0.0)
    if lam == 0.0:
        phi = dphi = 0.0
    else:
        theta_rel = wrap_angle(atan2(-dpy, -dpx) - spsi)
        phi, dphi = _phi_and_dphi(theta_rel, enc_type)
    rcc = rb * (1.0 + lam * phi)
    sp_factor = 1.0
    sp_dphi_world = 0.0
    stern = enc_type == "crossing_give_way" and STERN_PASS_LAMBDA > 0
    if stern:
        sp_diff = atan2(dpy, dpx) - obs.psi
        sp_cos = cos(sp_diff)
        if sp_cos > 0:
            sp_factor = 1.0 + STERN_PASS_LAMBDA * sp_cos
            sp_dphi_world = -STERN_PASS_LAMBDA * sin(sp_diff)
        rcc *= sp_factor
    h = d2 - rcc * rcc
    if not need_row:
        return h, None, None, None, None
    d = sqrt(d2) if d2 > 1e-12 else 1e-6
    vtx = obs.speed * cos(obs.psi)
    vty = obs.speed * sin(obs.psi)
    rcc_base = rb * (1.0 + lam * phi)
    dRdtheta = sp_factor * rb * lam * dphi
    if stern:
        dRdtheta += rcc_base * sp_dphi_world
    K = 2.0 * rcc * dRdtheta
    kq = K / (d2 if d2 > 1e-12 else 1e-12)
    ax = 2.0 * dpx + kq * dpy
    ay = 2.0 * dpy - kq * dpx
    b = -(ax * vtx + ay * vty)
    own_vx = su * cos(spsi)
    own_vy = su * sin(spsi)
    dvx = own_vx - vtx
    dvy = own_vy - vty
    v_close = -((dpx / d) * dvx + (dpy / d) * dvy)
    if v_close < 0.0:
        v_close = 0.0
    own_speed = sqrt(own_vx * own_vx + own_vy * own_vy)
    pf = 1.0 - d / (3.0 * rcc)
    proximity_tight = own_speed * (pf if pf > 0.0 else 0.0)
    tightening = GAMMA * (v_close * d + proximity_tight * rcc)
    c_hard = -ALPHA * h - b
    return h, ax, ay, c_hard + tightening, c_hard


def _qp2d_s(ux, uy, rows, v_max, tol=1e-9):
    """Scalar exact planar QP; rows = [(ax, ay, c)].  None if infeasible.
    Candidates and selection as in _qp2d."""
    ctol = 0.0
    for r in rows:
        ac = r[2] if r[2] >= 0.0 else -r[2]
        if ac > ctol:
            ctol = ac
    ctol = tol * (1.0 + ctol)
    vlim = v_max * (1.0 + 1e-9) + tol
    vlim2 = vlim * vlim
    ok = True
    if ux * ux + uy * uy > vlim2:
        ok = False
    else:
        for ax, ay, ci in rows:
            if ax * ux + ay * uy - ci < -ctol:
                ok = False
                break
    if ok:
        return ux, uy
    sqrt = math.sqrt
    cands = []
    nu = sqrt(ux * ux + uy * uy)
    cands.append((ux * v_max / nu, uy * v_max / nu) if nu > 1e-12 else (0.0, 0.0))
    n = len(rows)
    for i in range(n):
        ax, ay, ci = rows[i]
        aa = ax * ax + ay * ay
        if aa < 1e-18:
            continue
        k = (ci - (ax * ux + ay * uy)) / aa
        cands.append((ux + k * ax, uy + k * ay))
        px = ci / aa * ax
        py = ci / aa * ay
        rem = v_max * v_max - (px * px + py * py)
        if rem >= 0:
            na = sqrt(aa)
            rr = sqrt(rem)
            tx = -ay / na
            ty = ax / na
            cands.append((px + rr * tx, py + rr * ty))
            cands.append((px - rr * tx, py - rr * ty))
        for j in range(i + 1, n):
            bx, by, cj = rows[j]
            det = ax * by - ay * bx
            if det > 1e-12 or det < -1e-12:
                cands.append(((ci * by - ay * cj) / det, (ax * cj - ci * bx) / det))
    best = None
    bcost = float("inf")
    for x, y in cands:
        cost = (x - ux) * (x - ux) + (y - uy) * (y - uy)
        if cost >= bcost or x * x + y * y > vlim2:
            continue
        good = True
        for ax, ay, ci in rows:
            if ax * x + ay * y - ci < -ctol:
                good = False
                break
        if good:
            best = (x, y)
            bcost = cost
    return best


def _solve_s(ux, uy, rows4, v_max):
    """Constraint priority of Algorithm 1 on scalar rows (ax, ay, c, c_hard)."""
    full = [(r[0], r[1], r[2]) for r in rows4]
    u = _qp2d_s(ux, uy, full, v_max)
    if u is not None:
        return u
    # Barrier condition feasible: relax the tightening by the smallest
    # uniform factor (bisection on the retained fraction).
    hard = [(r[0], r[1], r[3]) for r in rows4]
    u = _qp2d_s(ux, uy, hard, v_max)
    if u is not None:
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            rows = [(r[0], r[1], r[3] + mid * (r[2] - r[3])) for r in rows4]
            if _qp2d_s(ux, uy, rows, v_max) is None:
                hi = mid
            else:
                lo = mid
        return _qp2d_s(ux, uy, [(r[0], r[1], r[3] + lo * (r[2] - r[3])) for r in rows4], v_max)
    # Barrier condition infeasible within the speed limit: least-violating
    # command (uniform normalised slack).
    norms = [math.sqrt(r[0] * r[0] + r[1] * r[1]) for r in rows4]
    lo = 0.0
    hi = max((r[3] + nm * v_max) / max(nm, 1e-12) for r, nm in zip(rows4, norms)) + 1.0
    best = None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cand = _qp2d_s(ux, uy, [(r[0], r[1], r[3] - mid * nm) for r, nm in zip(rows4, norms)], v_max)
        if cand is None:
            lo = mid
        else:
            hi, best = mid, cand
    if best is None:
        best = _qp2d_s(ux, uy, [(r[0], r[1], r[3] - hi * nm) for r, nm in zip(rows4, norms)], v_max)
    return best


# =====================================================================
# CCCBFController
# =====================================================================

class CCCBFController(BaseController):
    """
    CC-CBF controller with an exact planar QP.

    At each control update:
      1. Classify every detected target with the hysteretic classifier
         (switching signal tau, with give-way persistence until CPA).
      2. Evaluate h_CC and the affine CBF constraint for targets within
         ACTIVATION_RANGE.
      3. Shape the nominal velocity (Layer 2 heading bias or Layer 3 stern
         reference goal).
      4. Solve the QP exactly and convert the velocity into a heading and
         speed for the autopilot.

    Args:
        use_heading_bias: enable the Layer 2 heading bias (ablation switch).
        use_goal_shaping: enable the Layer 3 stern reference goal (ablation
            switch).
        name: display name.
    """

    def __init__(
        self,
        use_heading_bias: bool = True,
        use_goal_shaping: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "CC-CBF (Proposed)")
        self._risk: float = 0.0
        self._h_vals: Dict[str, float] = {}
        self._man: str = "hold_course"
        self._use_heading_bias: bool = use_heading_bias
        self._use_goal_shaping: bool = use_goal_shaping
        # Switching signal tau: last accepted class per target, used to
        # apply the classification dead-band of core.colregs (paper,
        # Remark "Classification hysteresis").
        self._tau: Dict[str, str] = {}

    # =================================================================
    # h_CC -- the barrier function
    # =================================================================

    @staticmethod
    def h_cc(ox, oy, opsi, tx, ty, tpsi, tl, tw, enc_type) -> float:
        """h_CC(x) = dist^2 - R_CC^2 (incl. stern-passage factor).  Positive => safe."""
        dx, dy = tx - ox, ty - oy
        d2 = dx * dx + dy * dy
        rb = _obs_rb(tl, tw)
        if d2 < 1e-12:
            return -(rb * rb)
        theta = wrap_angle(math.atan2(dy, dx) - opsi)
        phi, _ = _phi_and_dphi(theta, enc_type)
        lam = LAMBDA.get(enc_type, 0.0)
        rcc = rb * (1.0 + lam * phi)
        # Stern-passage factor for crossing_give_way (Rule 15)
        if enc_type == "crossing_give_way" and STERN_PASS_LAMBDA > 0:
            phi_world = math.atan2(oy - ty, ox - tx)  # bearing tgt -> own
            bow_cos = math.cos(phi_world - tpsi)       # 1 when own is ahead
            rcc *= (1.0 + STERN_PASS_LAMBDA * max(0.0, bow_cos))
        return d2 - rcc * rcc

    @staticmethod
    def r_cc_at_bearing(theta, enc_type, rb=R_BASE, tgt_psi=None,
                        own_psi=0.0):
        """Directional safety radius R_CC(theta) for visualisation.

        *theta* is the body-frame bearing own -> target.  When *tgt_psi* is
        given and the encounter is crossing_give_way, the stern-passage
        factor is included (*own_psi* converts theta to the world frame).
        """
        phi, _ = _phi_and_dphi(theta, enc_type)
        lam = LAMBDA.get(enc_type, 0.0)
        r = rb * (1.0 + lam * phi)
        if enc_type == "crossing_give_way" and STERN_PASS_LAMBDA > 0 and tgt_psi is not None:
            # world bearing tgt -> own = theta + own_psi + pi
            phi_world = wrap_angle(theta + own_psi + math.pi)
            sp_cos = math.cos(phi_world - tgt_psi)
            if sp_cos > 0:
                r *= (1.0 + STERN_PASS_LAMBDA * sp_cos)
        return r

    # =================================================================
    # Affine CBF constraint  a^T u >= c
    # =================================================================

    @staticmethod
    def _cbf_constraint(
        state: VesselState,
        obs: ObstacleState,
        enc: EncounterInfo,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Affine CBF constraint for one target.

            dh/dt = 2 (p_o - p_t)^T (v_o - v_t) - d(R_CC^2)/dt = a^T u + b

        Returns ``(a, c, h, c_hard)`` with the tightened right-hand side
        ``c = -alpha*h - b + tightening`` and the barrier condition
        ``c_hard = -alpha*h - b``.
        """
        # Relative position (own - obs)
        dpx = state.x - obs.x
        dpy = state.y - obs.y
        d2 = dpx * dpx + dpy * dpy
        d = math.sqrt(d2) if d2 > 1e-12 else 1e-6

        # Target velocity
        vtx = obs.speed * math.cos(obs.psi)
        vty = obs.speed * math.sin(obs.psi)

        enc_type = enc.encounter_type
        theta_rel = wrap_angle(math.atan2(-dpy, -dpx) - state.psi)
        rb = _obs_rb(obs.length, obs.width)
        lam = LAMBDA.get(enc_type, 0.0)
        phi, dphi = _phi_and_dphi(theta_rel, enc_type)
        rcc = rb * (1.0 + lam * phi)

        # Stern-passage factor for crossing_give_way (Rule 15): inflate the
        # barrier while the own ship is ahead of the target's bow.
        sp_factor = 1.0          # no modifier by default
        sp_dphi_world = 0.0      # derivative for the R_CC rate
        if enc_type == "crossing_give_way" and STERN_PASS_LAMBDA > 0:
            phi_world = math.atan2(dpy, dpx)  # bearing tgt -> own
            sp_diff = phi_world - obs.psi
            sp_cos = math.cos(sp_diff)
            if sp_cos > 0:
                sp_factor = 1.0 + STERN_PASS_LAMBDA * sp_cos
                sp_dphi_world = -STERN_PASS_LAMBDA * math.sin(sp_diff)
            rcc *= sp_factor

        # ---- Exact affine constraint (paper Sec. IV-C, Eq. (18)) ----
        # Within one control update the barrier orientation is frozen at the
        # measured heading (theta = beta - psi(t_k)), so theta_dot = beta_dot,
        # and beta_dot = (p_o - p_t)^perp . (v_o - v_t) / d^2 is linear in the
        # decision variable v_o.  With R_eff = R_dir * S (S = stern-passage
        # factor, whose world bearing phi_world also rotates at beta_dot):
        #   dR_eff^2/dt = K * (p_o - p_t)^perp . (v_o - v_t) / d^2,
        #   K = 2 R_eff (S rb lam dPhi + R_dir dS/dphi_world).
        # Hence dh/dt = a^T v_o + b with
        #   a = 2 (p_o - p_t) - (K / d^2) (p_o - p_t)^perp,   b = -a^T v_t.
        # Heading rotation between updates is not part of the certified
        # constraint; it is covered by the margin of Proposition 2.
        rcc_base = rb * (1.0 + lam * phi)  # R_dir (before the stern-passage factor)
        dRdtheta = sp_factor * rb * lam * dphi
        if enc_type == "crossing_give_way" and STERN_PASS_LAMBDA > 0:
            dRdtheta += rcc_base * sp_dphi_world
        K = 2.0 * rcc * dRdtheta
        d2s = max(d2, 1e-12)
        perp = np.array([-dpy, dpx])
        if CONSTRAINT_MODE == "exact":
            a = np.array([2.0 * dpx, 2.0 * dpy]) - (K / d2s) * perp
            b = -float(a @ np.array([vtx, vty]))
            if YAW_RATE_COMP:
                b += K * state.r
        else:  # legacy: radial normal, rate term at the current velocity
            ovx, ovy = state.u * math.cos(state.psi), state.u * math.sin(state.psi)
            bdot = (dpx * (ovy - vty) - dpy * (ovx - vtx)) / max(d2, 1.0)
            dR2 = 2.0 * rcc * (sp_factor * rb * lam * dphi * (bdot - state.r)
                               + ((rcc_base * sp_dphi_world * bdot)
                                  if (enc_type == "crossing_give_way" and STERN_PASS_LAMBDA > 0) else 0.0))
            a = np.array([2.0 * dpx, 2.0 * dpy])
            b = -2.0 * (dpx * vtx + dpy * vty) - dR2

        # Measured relative velocity (frozen before the solve) for tightening
        own_vx = state.u * math.cos(state.psi)
        own_vy = state.u * math.sin(state.psi)
        dvx = own_vx - vtx
        dvy = own_vy - vty

        h = d2 - rcc * rcc

        # Closing-speed tightening: v_close = max(0, -dp_hat . v_rel)
        dp_hat_x = dpx / d
        dp_hat_y = dpy / d
        v_close = max(0.0, -(dp_hat_x * dvx + dp_hat_y * dvy))

        # Proximity tightening for static / slow targets: tighten close to the
        # target even when v_close is small (prevents sliding along the
        # barrier boundary).
        own_speed = math.sqrt(own_vx * own_vx + own_vy * own_vy)
        proximity_tight = own_speed * max(0.0, 1.0 - d / (3.0 * rcc))

        tightening = GAMMA * (v_close * d + proximity_tight * rcc)
        c = -ALPHA * h - b + tightening
        c_hard = -ALPHA * h - b
        return a, c, h, c_hard

    # =================================================================
    # Exact QP with constraint priority (Algorithm 1)
    # =================================================================

    @staticmethod
    def _solve_cbf_qp(
        u_nom: np.ndarray,
        A: np.ndarray,
        c_vec: np.ndarray,
        v_max: float,
        c_hard: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Exact solution of the planar CBF-QP

            min ||u - u_nom||^2   s.t.  A[i] @ u >= c[i],  ||u|| <= v_max

        by active-set enumeration.

        Priority when the tightened set is empty (paper Sec. V, Algorithm 1): the
        barrier condition  A u >= c_hard  (= -alpha h - b) is hard; the
        anticipatory tightening  c - c_hard >= 0  is relaxed by the smallest
        uniform factor that restores feasibility.  If even the barrier
        condition cannot be met within the speed limit, the least-violating
        command (uniform normalised slack) is returned.
        """
        n = A.shape[0] if A.ndim == 2 else 0
        if n == 0:
            norm = np.linalg.norm(u_nom)
            return u_nom * (v_max / norm) if norm > v_max else u_nom.copy()
        if c_hard is None:
            c_hard = c_vec
        u = _qp2d(u_nom, A, c_vec, v_max)
        if u is not None:
            return u
        u = _qp2d(u_nom, A, c_hard, v_max)
        if u is not None:
            lo, hi = 0.0, 1.0            # largest feasible tightening fraction
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if _qp2d(u_nom, A, c_hard + mid * (c_vec - c_hard), v_max) is None:
                    hi = mid
                else:
                    lo = mid
            return _qp2d(u_nom, A, c_hard + lo * (c_vec - c_hard), v_max)
        norms = np.linalg.norm(A, axis=1)
        lo, hi = 0.0, float(np.max((c_hard + norms * v_max) / np.maximum(norms, 1e-12))) + 1.0
        best = None
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cand = _qp2d(u_nom, A, c_hard - mid * norms, v_max)
            if cand is None:
                lo = mid
            else:
                hi, best = mid, cand
        if best is None:
            best = _qp2d(u_nom, A, c_hard - hi * norms, v_max)
        return best

    # =================================================================
    # Control step
    # =================================================================

    def compute_command(
        self,
        state: VesselState,
        obstacles: List[ObstacleState],
        encounters: List[EncounterInfo],
        goal_x: float,
        goal_y: float,
        time: float,
        dt: float,
        scenario_config: Optional[ScenarioConfig] = None,
    ) -> ControlCommand:
        # The scalar path implements the default constraint form only; the
        # reference path below also honours CONSTRAINT_MODE / YAW_RATE_COMP.
        if FAST_STEP and CONSTRAINT_MODE == "exact" and not YAW_RATE_COMP:
            tau_prev = self._tau
            types = [classify_type_fast(state, obs, tau_prev.get(obs.label)) for obs in obstacles]
            if PERSIST_GIVE_WAY:
                for j, obs in enumerate(obstacles):
                    prev = tau_prev.get(obs.label)
                    if prev in _GIVE_WAY and types[j] != prev and _closing(state, obs):
                        types[j] = prev
            self._tau = {obs.label: t for obs, t in zip(obstacles, types)}
            nom_heading = self.nominal_heading(state, goal_x, goal_y)
            nom_speed = cfg.CRUISE_SPEED
            if not obstacles:
                self._risk = 0.0
                self._man = "hold_course"
                return ControlCommand(desired_heading=nom_heading, desired_speed=nom_speed,
                                      manoeuvre="hold_course", explanation="CC-CBF: no obstacles")
            return self._fast_tail(state, obstacles, types, nom_heading, nom_speed)

        # ---- NumPy reference path ----
        # Build the hysteretic switching signal tau from the raw geometry.
        # The simulator's memoryless classification is shared with the
        # baselines and the metrics; CC-CBF's barrier uses its own
        # dead-band class so that tau has a dwell time.
        encounters = [classify_encounter(state, obs,
                                         prev_type=self._tau.get(obs.label),
                                         light=True)
                      for obs in obstacles]
        if PERSIST_GIVE_WAY:
            # Give-way obligations (Rules 13-15) persist until the target is
            # past CPA (range opening), as stated in paper Sec. III-D.
            for j, (obs, enc) in enumerate(zip(obstacles, encounters)):
                prev = self._tau.get(obs.label)
                if (prev in _GIVE_WAY and enc.encounter_type != prev
                        and _closing(state, obs)):
                    encounters[j] = EncounterInfo(encounter_type=prev)
        self._tau = {obs.label: enc.encounter_type
                     for obs, enc in zip(obstacles, encounters)}

        # Nominal: head toward the goal at cruise speed
        nom_heading = self.nominal_heading(state, goal_x, goal_y)
        nom_speed = cfg.CRUISE_SPEED

        if not obstacles:
            self._risk = 0.0
            self._man = "hold_course"
            return ControlCommand(
                desired_heading=nom_heading,
                desired_speed=nom_speed,
                manoeuvre="hold_course",
                explanation="CC-CBF: no obstacles",
            )

        # Build CBF constraints for targets within the activation range
        A_rows = []
        c_rows = []
        ch_rows = []
        h_min = float("inf")
        # Most urgent give-way encounter (drives the nominal shaping)
        closest_gw_enc = None
        closest_gw_dist = float("inf")

        for obs, enc in zip(obstacles, encounters):
            d = math.hypot(state.x - obs.x, state.y - obs.y)
            a, c, h, ch = self._cbf_constraint(state, obs, enc)
            self._h_vals[obs.label] = h
            if h < h_min:
                h_min = h

            if d <= ACTIVATION_RANGE:
                A_rows.append(a)
                c_rows.append(c)
                ch_rows.append(ch)

            if enc.encounter_type in _GIVE_WAY \
                    and (not NOMINAL_CPA_GATE or _not_clear(state, obs)):
                if d < closest_gw_dist:
                    closest_gw_dist = d
                    closest_gw_enc = enc

        self._risk = min(1.0, math.exp(-h_min / max(R_BASE**2 * 0.5, 1.0)))

        if not A_rows:
            self._man = "hold_course"
            return ControlCommand(
                desired_heading=nom_heading,
                desired_speed=nom_speed,
                manoeuvre="hold_course",
                explanation="CC-CBF: safe",
            )

        # ---- Nominal shaping (Layers 2 and 3) ----
        biased_heading = nom_heading
        biased_speed = nom_speed
        if closest_gw_enc is not None and closest_gw_dist < ACTIVATION_RANGE:
            enc_type = closest_gw_enc.encounter_type
            proximity_factor = max(0.0, 1.0 - closest_gw_dist / ACTIVATION_RANGE)

            if enc_type == "crossing_give_way" and self._use_goal_shaping:
                # Layer 3: steer towards a point astern of the target at
                # reduced speed so that the target crosses first (Rule 15).
                obs_gw = None
                for obs, enc in zip(obstacles, encounters):
                    if enc.encounter_type == "crossing_give_way" and \
                            (not NOMINAL_CPA_GATE or _not_clear(state, obs)):
                        obs_gw = obs
                        break
                if obs_gw is None:
                    obs_gw = obstacles[0]

                stern_x = obs_gw.x - STERN_OFFSET * math.cos(obs_gw.psi)
                stern_y = obs_gw.y - STERN_OFFSET * math.sin(obs_gw.psi)
                biased_heading = math.atan2(stern_y - state.y,
                                            stern_x - state.x)
                biased_speed = nom_speed * STERN_SPEED_FACTOR

            elif enc_type != "crossing_give_way" and self._use_heading_bias:
                # Layer 2: starboard heading bias (head-on / overtaking)
                bias_deg = BIAS_DEG.get(enc_type, 25.0)
                bias_angle = -math.radians(bias_deg) * (
                    1.0 if BIAS_PROFILE == "full" else proximity_factor)
                biased_heading = wrap_angle(nom_heading + bias_angle)

        u_nom = np.array([biased_speed * math.cos(biased_heading),
                          biased_speed * math.sin(biased_heading)])

        u_safe = self._solve_cbf_qp(u_nom, np.array(A_rows), np.array(c_rows),
                                    cfg.MAX_SPEED_MPS, np.array(ch_rows))
        des_heading = math.atan2(u_safe[1], u_safe[0])
        des_speed = np.linalg.norm(u_safe)
        des_speed = max(0.0, min(des_speed, cfg.MAX_SPEED_MPS))

        man = manoeuvre_label(wrap_angle(des_heading - nom_heading),
                              des_speed / max(nom_speed, 0.01))
        self._man = man
        return ControlCommand(
            desired_heading=des_heading,
            desired_speed=des_speed,
            manoeuvre=man,
            explanation="CC-CBF: safe" if man == "hold_course" else "CC-CBF: " + man,
        )

    def _fast_tail(self, state, obstacles, types, nom_heading, nom_speed):
        """Scalar implementation of the constraint assembly, nominal shaping
        and exact QP (same mathematics as the reference path)."""
        sx, sy, spsi, su = state.x, state.y, state.psi, state.u
        rows4 = []
        h_min = float("inf")
        closest_gw_type = None
        closest_gw_dist = float("inf")
        hv = self._h_vals
        for obs, et in zip(obstacles, types):
            dx = sx - obs.x
            dy = sy - obs.y
            d2 = dx * dx + dy * dy
            d = math.hypot(dx, dy)
            active = d <= ACTIVATION_RANGE
            h, ax, ay, c, ch = _row_scalar(sx, sy, spsi, su, obs, et, active, dx, dy, d2)
            hv[obs.label] = h
            if h < h_min:
                h_min = h
            if active:
                rows4.append((ax, ay, c, ch))
            if et in _GIVE_WAY and (not NOMINAL_CPA_GATE or _not_clear(state, obs)):
                if d < closest_gw_dist:
                    closest_gw_dist = d
                    closest_gw_type = et
        self._risk = min(1.0, math.exp(-h_min / max(R_BASE ** 2 * 0.5, 1.0)))
        if not rows4:
            self._man = "hold_course"
            return ControlCommand(desired_heading=nom_heading, desired_speed=nom_speed,
                                  manoeuvre="hold_course", explanation="CC-CBF: safe")
        biased_heading = nom_heading
        biased_speed = nom_speed
        if closest_gw_type is not None and closest_gw_dist < ACTIVATION_RANGE:
            if closest_gw_type == "crossing_give_way" and self._use_goal_shaping:
                obs_gw = None
                for obs, et in zip(obstacles, types):
                    if et == "crossing_give_way" and \
                            (not NOMINAL_CPA_GATE or _not_clear(state, obs)):
                        obs_gw = obs
                        break
                if obs_gw is None:
                    obs_gw = obstacles[0]
                stern_x = obs_gw.x - STERN_OFFSET * math.cos(obs_gw.psi)
                stern_y = obs_gw.y - STERN_OFFSET * math.sin(obs_gw.psi)
                biased_heading = math.atan2(stern_y - sy, stern_x - sx)
                biased_speed = nom_speed * STERN_SPEED_FACTOR
            elif closest_gw_type != "crossing_give_way" and self._use_heading_bias:
                proximity_factor = max(0.0, 1.0 - closest_gw_dist / ACTIVATION_RANGE)
                bias_angle = -math.radians(BIAS_DEG.get(closest_gw_type, 25.0)) * (
                    1.0 if BIAS_PROFILE == "full" else proximity_factor)
                biased_heading = wrap_angle(nom_heading + bias_angle)
        ux = biased_speed * math.cos(biased_heading)
        uy = biased_speed * math.sin(biased_heading)
        vx, vy = _solve_s(ux, uy, rows4, cfg.MAX_SPEED_MPS)
        des_heading = math.atan2(vy, vx)
        des_speed = max(0.0, min(math.sqrt(vx * vx + vy * vy), cfg.MAX_SPEED_MPS))
        man = manoeuvre_label(wrap_angle(des_heading - nom_heading), des_speed / max(nom_speed, 0.01))
        self._man = man
        return ControlCommand(desired_heading=des_heading, desired_speed=des_speed, manoeuvre=man,
                              explanation="CC-CBF: safe" if man == "hold_course" else "CC-CBF: " + man)

    # =================================================================
    # Introspection
    # =================================================================

    def get_scalar_risk(self) -> float:
        return self._risk

    def get_h_values(self) -> Dict[str, float]:
        """Latest h_CC value per target label."""
        return dict(self._h_vals)

    def get_manoeuvre(self) -> str:
        return self._man

    def reset(self) -> None:
        super().reset()
        self._risk = 0.0
        self._h_vals.clear()
        self._tau.clear()
        self._man = "hold_course"
