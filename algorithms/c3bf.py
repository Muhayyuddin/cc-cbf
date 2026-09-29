"""
Baseline B1: Collision-Cone CBF (C3BF) -- Tayal et al. (ACC 2024)
===============================================================
Implements the vanilla Collision Cone Control Barrier Function from:

    Tayal, Singh, Keshavan & Kolathaya,
    "Control Barrier Functions in Dynamic UAVs for Kinematic Obstacle
     Avoidance: A Collision Cone Approach", ACC 2024 / arXiv:2303.15871

Core barrier:
    h(x,t) = <p_rel, v_rel> + ||p_rel|| * ||v_rel|| * cos(phi)

where
    p_rel = p_obs - p_own          (relative position)
    v_rel = v_obs - v_own          (relative velocity, obstacle minus own)
    phi   = half-angle of collision cone
    cos(phi) = sqrt(||p_rel||^2 - r^2) / ||p_rel||
    r     = combined safety radius

The barrier is linearised in u = [vx, vy] (own velocity) and the
resulting constraints are enforced by cyclic projection.

Differences from CC-CBF:
  1. no directional asymmetry (isotropic radius, no lambda / theta_C);
  2. no COLREG shaping of the nominal velocity;
  => pure safety, no COLREG awareness.

This is the closest CBF-class baseline: the same velocity-space CBF
structure and tightening, without the COLREG extensions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import data.config as cfg
from algorithms.base_controller import (
    BaseController, cyclic_projection, manoeuvre_label,
)
from core.entities import (
    ControlCommand, EncounterInfo, ObstacleState, ScenarioConfig, VesselState,
)
from core.geometry import wrap_angle

# =====================================================================
# C3BF parameters
# =====================================================================

_OWN_HD = math.hypot(cfg.USV_LENGTH, cfg.USV_WIDTH) / 2.0
_OBS_HD = math.hypot(cfg.DEFAULT_TARGET_LENGTH,
                     cfg.DEFAULT_TARGET_WIDTH) / 2.0
R_SAFE = _OWN_HD + _OBS_HD + cfg.SAFETY_BUFFER  # isotropic radius

# CBF class-K gain
ALPHA = 0.8

# Velocity-dependent tightening gain (same idea as CC-CBF for fair comparison)
GAMMA = 1.0

# Activation range
ACTIVATION_RANGE = 95.0  # m


def _obs_radius(length: float, width: float) -> float:
    """Combined safety radius for a specific obstacle."""
    return _OWN_HD + math.hypot(length, width) / 2.0 + cfg.SAFETY_BUFFER


# =====================================================================
# C3BFController
# =====================================================================

class C3BFController(BaseController):
    """
    Collision Cone CBF (Tayal et al.) adapted for maritime USV.

    Uses the isotropic collision-cone barrier:
        h = <p_rel, v_rel> + ||p_rel|| ||v_rel|| cos(phi)

    No COLREG awareness — purely geometric safety.
    """

    def __init__(self) -> None:
        super().__init__(name="C3BF (Tayal)")
        self._risk: float = 0.0
        self._h_vals: Dict[str, float] = {}
        self._man: str = "hold_course"

    # =================================================================
    # h — the vanilla collision-cone barrier
    # =================================================================

    @staticmethod
    def h_c3bf(
        own_x: float, own_y: float, own_vx: float, own_vy: float,
        obs_x: float, obs_y: float, obs_vx: float, obs_vy: float,
        r: float,
    ) -> float:
        """
        h(x,t) = <p_rel, v_rel> + ||p_rel|| * ||v_rel|| * cos(phi)

        where cos(phi) = sqrt(||p_rel||^2 - r^2) / ||p_rel||
        Positive => relative velocity is outside the collision cone => safe.
        """
        # Relative position: obstacle - own
        prx = obs_x - own_x
        pry = obs_y - own_y
        p_norm_sq = prx * prx + pry * pry
        p_norm = math.sqrt(p_norm_sq) if p_norm_sq > 1e-12 else 1e-6

        # Relative velocity: obstacle - own  (convention from Tayal et al.)
        vrx = obs_vx - own_vx
        vry = obs_vy - own_vy
        v_norm = math.sqrt(vrx * vrx + vry * vry)

        # Dot product <p_rel, v_rel>
        dot_pv = prx * vrx + pry * vry

        # cos(phi) = sqrt(||p||^2 - r^2) / ||p||
        inner = max(0.0, p_norm_sq - r * r)
        cos_phi = math.sqrt(inner) / p_norm

        h = dot_pv + p_norm * v_norm * cos_phi
        return h

    # =================================================================
    # CBF constraint:  a^T u >= c  (linear in u = [vx_own, vy_own])
    # =================================================================

    @staticmethod
    def _cbf_constraint(
        state: VesselState,
        obs: ObstacleState,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Linear constraint a^T u >= c for one obstacle.

        h depends on u through v_rel = v_obs - u.  Linearising h around the
        current velocity u0,

            h(u) ~ h(u0) + (dh/du)^T (u - u0),

        and requiring h(u) >= -alpha * h(u0) gives

            a = dh/du,   c = -(1 + alpha) h(u0) + a^T u0 + tightening.

        Returns ``(a, c, h)``.
        """
        # Own velocity
        own_vx = state.u * math.cos(state.psi)
        own_vy = state.u * math.sin(state.psi)

        # Obstacle velocity
        obs_vx = obs.speed * math.cos(obs.psi)
        obs_vy = obs.speed * math.sin(obs.psi)

        r = _obs_radius(obs.length, obs.width)

        # Relative position: obs - own
        prx = obs.x - state.x
        pry = obs.y - state.y
        p_norm_sq = prx * prx + pry * pry
        p_norm = math.sqrt(p_norm_sq) if p_norm_sq > 1e-12 else 1e-6

        # Relative velocity: obs - own
        vrx = obs_vx - own_vx
        vry = obs_vy - own_vy
        v_norm_sq = vrx * vrx + vry * vry
        v_norm = math.sqrt(v_norm_sq) if v_norm_sq > 1e-12 else 1e-6

        # cos(phi)
        inner = max(0.0, p_norm_sq - r * r)
        cos_phi = math.sqrt(inner) / p_norm

        # h value
        dot_pv = prx * vrx + pry * vry
        h = dot_pv + p_norm * v_norm * cos_phi

        # ---- Gradient dh/du ----
        # h = <p, v_obs - u> + ||p|| * ||v_obs - u|| * cos_phi
        # dh/du = -p + ||p|| * cos_phi * d||v_rel||/du
        # d||v_rel||/du = d||v_obs - u||/du = -(v_obs - u)/||v_obs - u|| = -v_rel/||v_rel||

        # dh/du = -p + ||p|| * cos_phi * (-v_rel / ||v_rel||)
        #       = -p - ||p|| * cos_phi * v_rel / ||v_rel||
        dh_du_x = -prx - p_norm * cos_phi * vrx / v_norm
        dh_du_y = -pry - p_norm * cos_phi * vry / v_norm

        # Linearisation around the current velocity (see docstring)
        a = np.array([dh_du_x, dh_du_y])
        u_current = np.array([own_vx, own_vy])
        bias = a @ u_current

        # ---- Tightening (velocity-dependent, same as CC-CBF for fairness) ----
        dpx = state.x - obs.x  # own - obs
        dpy = state.y - obs.y
        d = math.sqrt(dpx * dpx + dpy * dpy) if (dpx * dpx + dpy * dpy) > 1e-12 else 1e-6
        dp_hat_x = dpx / d
        dp_hat_y = dpy / d
        dvx = own_vx - obs_vx
        dvy = own_vy - obs_vy
        v_close = max(0.0, -(dp_hat_x * dvx + dp_hat_y * dvy))

        own_speed = math.sqrt(own_vx * own_vx + own_vy * own_vy)
        proximity_tight = own_speed * max(0.0, 1.0 - d / (3.0 * r))
        tightening = GAMMA * (v_close * d + proximity_tight * r)

        c = -ALPHA * h - h + bias + tightening
        # Simplifies to: a^T u >= -(1+alpha)*h + a^T u_current + tightening

        return a, c, h

    # =================================================================
    # QP:  min ||u - u_nom||^2  s.t.  A u >= c, ||u|| <= v_max
    # =================================================================

    @staticmethod
    def _solve_cbf_qp(
        u_nom: np.ndarray,
        A: np.ndarray,
        c_vec: np.ndarray,
        v_max: float,
    ) -> np.ndarray:
        """Cyclic projection (10 sweeps) followed by speed clipping."""
        return cyclic_projection(u_nom, A, c_vec, v_max, max_iter=10)

    # =================================================================
    # Main control loop
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

        nom_heading = self.nominal_heading(state, goal_x, goal_y)
        nom_speed = cfg.CRUISE_SPEED

        if not obstacles:
            self._risk = 0.0
            self._man = "hold_course"
            return ControlCommand(
                desired_heading=nom_heading,
                desired_speed=nom_speed,
                manoeuvre="hold_course",
                explanation="C3BF: no obstacles",
            )

        # Build constraints
        A_rows = []
        c_rows = []
        h_min = float("inf")

        for obs in obstacles:
            d = math.hypot(state.x - obs.x, state.y - obs.y)
            a, c, h = self._cbf_constraint(state, obs)
            self._h_vals[obs.label] = h
            if h < h_min:
                h_min = h
            if d <= ACTIVATION_RANGE:
                A_rows.append(a)
                c_rows.append(c)

        self._risk = min(1.0, math.exp(-h_min / max(R_SAFE**2 * 0.5, 1.0)))

        if not A_rows:
            self._man = "hold_course"
            return ControlCommand(
                desired_heading=nom_heading,
                desired_speed=nom_speed,
                manoeuvre="hold_course",
                explanation="C3BF: safe (h={:.0f})".format(h_min),
            )

        # NO COLREG bias — this is vanilla C3BF
        u_nom = np.array([nom_speed * math.cos(nom_heading),
                          nom_speed * math.sin(nom_heading)])

        A = np.array(A_rows)
        c_vec = np.array(c_rows)

        u_safe = self._solve_cbf_qp(u_nom, A, c_vec, cfg.MAX_SPEED_MPS)

        des_heading = math.atan2(u_safe[1], u_safe[0])
        des_speed = float(np.linalg.norm(u_safe))
        des_speed = max(0.0, min(des_speed, cfg.MAX_SPEED_MPS))

        heading_delta = wrap_angle(des_heading - nom_heading)
        speed_ratio = des_speed / max(nom_speed, 0.01)
        man = manoeuvre_label(heading_delta, speed_ratio)
        self._man = man

        if man == "hold_course":
            expl = "C3BF: safe (h={:.1f})".format(h_min)
        else:
            expl = "C3BF: {} (h={:.1f}, r={:.2f})".format(man, h_min, self._risk)

        return ControlCommand(
            desired_heading=des_heading,
            desired_speed=des_speed,
            manoeuvre=man,
            explanation=expl,
        )

    # =================================================================
    # Interface
    # =================================================================

    def get_scalar_risk(self) -> float:
        return self._risk

    def get_h_values(self) -> Dict[str, float]:
        return dict(self._h_vals)

    def get_manoeuvre(self) -> str:
        return self._man

    def reset(self) -> None:
        super().reset()
        self._risk = 0.0
        self._h_vals.clear()
        self._man = "hold_course"

