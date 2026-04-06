"""
Baseline: Turning-Circle CBF (TC-CBF) — Lee et al. (IEEE Access, 2025)
========================================================================
Implements the core concept of:

    Lee et al., "Turning-Circle Control Barrier Functions for
    COLREG-Compliant Maritime Collision Avoidance," IEEE Access 2025.

Key idea:
    For each obstacle, construct TWO barrier functions — one for a port
    avoidance and one for a starboard avoidance — based on the vessel's
    turning circle geometry.  External switching logic selects which
    barrier to enforce based on the COLREG encounter type.

Barrier construction (per obstacle, per turn direction):
    The turning circle of radius R_turn is tangent to the vessel's
    current heading.  The center of the starboard turning circle is:
        c_stbd = p_own + R_turn * n_stbd
    where n_stbd is the unit vector 90° clockwise from the heading.
    Similarly for port.

    The barrier function ensures the obstacle doesn't enter the
    turning-circle sweep region:
        h_dir(x) = ||p_obs - c_dir||^2 - (R_turn + r_safe)^2

    where r_safe is the combined safety radius.

    The CBF condition  ḣ >= -α·h  yields a linear constraint in the
    velocity control input, solved via the same QP framework.

Key differences from CC-CBF:
    - Uses TWO barriers per obstacle (port + starboard) instead of ONE
    - Requires EXTERNAL switching logic to select active barrier
    - Turning-circle geometry ties the barrier to the vessel's kinematics
    - During encounter transitions, switching can cause discontinuities

This implementation is a faithful reconstruction of the core concept
adapted to our simulation framework for fair comparison.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithms.base_controller import BaseController
from core.entities import (
    ControlCommand, EncounterInfo, ObstacleState, RiskVector,
    ScenarioConfig, USVParameters, VesselState,
)
from core.geometry import wrap_angle
import data.config as cfg

# =====================================================================
# TC-CBF parameters
# =====================================================================

_OWN_HD = math.hypot(cfg.USV_LENGTH, cfg.USV_WIDTH) / 2.0
_OBS_HD = math.hypot(cfg.DEFAULT_TARGET_LENGTH,
                     cfg.DEFAULT_TARGET_WIDTH) / 2.0
R_SAFE_BASE = _OWN_HD + _OBS_HD + cfg.SAFETY_BUFFER

# Turning radius — derived from the vessel's minimum turning circle.
# For the MBZIRC USV:  R_turn ≈ v_max^2 / (r_max * v_max) ≈ v / r_max
# At cruise speed ~3 m/s with max yaw rate ~1 rad/s (from N_r/I_z damping):
# R_turn ≈ 3/1 ≈ 3 m.  For safety, we use a larger effective radius
# that accounts for the vessel footprint and dynamic effects.
# Lee et al. use R_turn proportional to speed; we fix at a representative value.
R_TURN = 8.0  # m, effective turning radius at cruise speed

# CBF class-K gain  (Lee et al. use a simple linear class-K function)
ALPHA = 0.5

# Velocity-dependent tightening gain — Lee et al. do NOT use the
# aggressive proximity tightening that CC-CBF uses; we keep only a
# mild closing-speed term for numerical robustness.
GAMMA = 1.0

# Activation range
ACTIVATION_RANGE = 95.0  # m

# COLREG switching table: maps encounter type to preferred turn direction
# "stbd" = starboard (right) turn, "port" = port (left) turn
COLREG_TURN_DIR: Dict[str, str] = {
    "head_on":           "stbd",   # Rule 14: alter course to starboard
    "crossing_give_way": "stbd",   # Rule 15: keep clear, pass astern (starboard)
    "crossing_stand_on": "stbd",   # Rule 17: maintain course, but if must act → stbd
    "overtaking":        "stbd",   # Rule 13: pass on starboard side
    "none":              "stbd",   # Default: starboard preference
}

# Nominal heading bias — Lee et al. rely ONLY on the turning-circle
# barrier geometry for COLREG compliance.  Unlike CC-CBF, there is NO
# directional bias on the nominal heading.  We include a very small
# bias (5°) solely to break symmetry in perfectly symmetric ICs.
BIAS_DEG: Dict[str, float] = {
    "head_on":           5.0,
    "crossing_give_way": 5.0,
    "overtaking":        5.0,
}


def _obs_radius(length: float, width: float) -> float:
    """Combined safety radius for a specific obstacle."""
    return _OWN_HD + math.hypot(length, width) / 2.0 + cfg.SAFETY_BUFFER


# =====================================================================
# TurningCircleCBFController
# =====================================================================

class TurningCircleCBFController(BaseController):
    """
    Turning-Circle CBF (Lee et al., IEEE Access 2025).

    Constructs port and starboard turning-circle barriers for each
    obstacle.  COLREG switching logic selects which barrier to enforce.
    The QP minimises deviation from the nominal velocity subject to
    the selected barrier constraints.

    Key architectural difference from CC-CBF:
      - Two barriers per obstacle (only one active via switching)
      - Barrier geometry depends on turning circle, not distance
      - External encounter-type switching selects active constraint
    """

    def __init__(self) -> None:
        super().__init__(name="TC-CBF (Lee)")
        self._risk: float = 0.0
        self._h_vals: Dict[str, float] = {}
        self._man: str = "hold_course"
        # Track active turn direction per obstacle for hysteresis
        self._active_dir: Dict[str, str] = {}

    # =================================================================
    # Turning-circle center computation
    # =================================================================

    @staticmethod
    def _turning_center(
        px: float, py: float, psi: float,
        r_turn: float, direction: str
    ) -> Tuple[float, float]:
        """
        Compute the center of the turning circle.

        For starboard turn: center is 90° clockwise from heading
            c = p + R_turn * [cos(psi - π/2), sin(psi - π/2)]
              = p + R_turn * [sin(psi), -cos(psi)]    ... wait, that's port
        Actually:
            Starboard (right) turn → center to starboard of vessel
            n_stbd = [sin(psi), -cos(psi)]  ... no, let's be careful.

        Heading psi: vessel points in direction [cos(psi), sin(psi)]
        Starboard is 90° clockwise: rotate heading by -90°
            n_stbd = [cos(psi - π/2), sin(psi - π/2)]
                   = [sin(psi), -cos(psi)]
        Port is 90° counter-clockwise: rotate heading by +90°
            n_port = [cos(psi + π/2), sin(psi + π/2)]
                   = [-sin(psi), cos(psi)]
        """
        if direction == "stbd":
            # Center to starboard
            cx = px + r_turn * math.sin(psi)
            cy = py - r_turn * math.cos(psi)
        else:
            # Center to port
            cx = px - r_turn * math.sin(psi)
            cy = py + r_turn * math.cos(psi)
        return cx, cy

    # =================================================================
    # h_TC — the turning-circle barrier function
    # =================================================================

    @staticmethod
    def h_tc(
        own_x: float, own_y: float, own_psi: float,
        obs_x: float, obs_y: float,
        obs_length: float, obs_width: float,
        r_turn: float, direction: str,
    ) -> float:
        """
        Turning-circle barrier:
            h_dir(x) = ||p_obs - c_dir||^2 - (R_turn + r_safe)^2

        Positive means the obstacle is outside the turning circle
        sweep region → safe.
        """
        cx, cy = TurningCircleCBFController._turning_center(
            own_x, own_y, own_psi, r_turn, direction
        )
        r_safe = _obs_radius(obs_length, obs_width)
        dx = obs_x - cx
        dy = obs_y - cy
        d2 = dx * dx + dy * dy
        r_total = r_turn + r_safe
        return d2 - r_total * r_total

    # =================================================================
    # CBF constraint: a^T u >= c
    # =================================================================

    @staticmethod
    def _cbf_constraint(
        state: VesselState,
        obs: ObstacleState,
        enc: EncounterInfo,
        direction: str,
        r_turn: float = R_TURN,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute the linear CBF constraint for the turning-circle barrier.

        The turning-circle center moves with the ownship:
            c = p_own + R_turn * n(psi, direction)

        Time derivative:
            ḣ = 2(p_obs - c)^T * (v_obs - ċ)

        where ċ = v_own + R_turn * ṅ(psi, r)
        ṅ depends on yaw rate r, which is a state (not control).

        Under the velocity-controlled abstraction (same as CC-CBF),
        v_own = u is the control input. The ṅ term (from yaw rate)
        is treated as a known bias.

        ḣ = 2(p_obs - c)^T * v_obs - 2(p_obs - c)^T * v_own
            - 2(p_obs - c)^T * R_turn * ṅ

        The constraint ḣ >= -α·h becomes:
            -2(p_obs - c)^T * u >= -α·h - [2(p_obs - c)^T * v_obs
                                            - 2(p_obs - c)^T * R_turn * ṅ]
        i.e.:
            a^T u >= c_val
        where a = -2(p_obs - c)  [note the sign: we want to move c away from obs]

        Actually more carefully: let d_vec = p_obs - c.
            ḣ = d(||d_vec||^2)/dt = 2 d_vec^T * ḋ_vec
            ḋ_vec = v_obs - ċ = v_obs - (u + R_turn * ṅ)

        So:
            ḣ = 2 d_vec^T * (v_obs - u - R_turn * ṅ)
              = -2 d_vec^T * u + 2 d_vec^T * (v_obs - R_turn * ṅ)

        => a = -2 d_vec,  b = 2 d_vec^T * (v_obs - R_turn * ṅ)
        => constraint: a^T u >= -α h - b
        """
        # Turning-circle center
        cx, cy = TurningCircleCBFController._turning_center(
            state.x, state.y, state.psi, r_turn, direction
        )

        # d_vec = p_obs - c
        dx = obs.x - cx
        dy = obs.y - cy
        d2 = dx * dx + dy * dy
        d = math.sqrt(d2) if d2 > 1e-12 else 1e-6

        # a = -2 * d_vec  (negative because we want u to increase ||d_vec||)
        a = np.array([-2.0 * dx, -2.0 * dy])

        # Target velocity
        vtx = obs.speed * math.cos(obs.psi)
        vty = obs.speed * math.sin(obs.psi)

        # ṅ — derivative of the normal vector w.r.t. yaw rate
        # For stbd: n = [sin(psi), -cos(psi)]
        #   dn/dt = r * [cos(psi), sin(psi)]  (= r * heading_vector)
        # For port: n = [-sin(psi), cos(psi)]
        #   dn/dt = r * [-cos(psi), -sin(psi)] = -r * heading_vector
        r_yaw = state.r  # current yaw rate
        hx = math.cos(state.psi)
        hy = math.sin(state.psi)
        if direction == "stbd":
            ndot_x = r_yaw * hx
            ndot_y = r_yaw * hy
        else:
            ndot_x = -r_yaw * hx
            ndot_y = -r_yaw * hy

        # b = 2 d_vec^T * (v_obs - R_turn * ṅ)
        b_x = vtx - r_turn * ndot_x
        b_y = vty - r_turn * ndot_y
        b = 2.0 * (dx * b_x + dy * b_y)

        # h value
        r_safe = _obs_radius(obs.length, obs.width)
        r_total = r_turn + r_safe
        h = d2 - r_total * r_total

        # Closing speed tightening (mild — just closing speed, no
        # proximity term; Lee et al. use a basic class-K function)
        own_vx = state.u * math.cos(state.psi)
        own_vy = state.u * math.sin(state.psi)
        # Relative velocity between obstacle and turning center
        vc_x = own_vx + r_turn * ndot_x  # turning center velocity
        vc_y = own_vy + r_turn * ndot_y
        dvx = vc_x - vtx
        dvy = vc_y - vty
        d_hat_x = dx / d
        d_hat_y = dy / d
        v_close = max(0.0, -(d_hat_x * dvx + d_hat_y * dvy))

        tightening = GAMMA * v_close * d
        c_val = -ALPHA * h - b + tightening

        return a, c_val, h

    # =================================================================
    # Backup distance-based CBF constraint (ensures basic safety)
    # =================================================================

    @staticmethod
    def _distance_cbf_constraint(
        state: VesselState,
        obs: ObstacleState,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Standard isotropic distance-based CBF as a **backup** safety net:
            h_dist = ||p_o - p_t||^2 - R_safe^2

        This is the key safety mechanism for TC-CBF.  The turning-circle
        barrier provides COLREG directionality but has geometric blind
        spots.  This backup ensures no collision regardless of geometry.
        """
        ALPHA_BACKUP = 0.8
        GAMMA_BACKUP = 4.0

        dpx = state.x - obs.x
        dpy = state.y - obs.y
        d2 = dpx * dpx + dpy * dpy
        d = math.sqrt(d2) if d2 > 1e-12 else 1e-6

        r_safe = _obs_radius(obs.length, obs.width)

        # a = 2(p_o - p_t)
        a = np.array([2.0 * dpx, 2.0 * dpy])

        # b = -2(p_o - p_t)^T v_t
        vtx = obs.speed * math.cos(obs.psi)
        vty = obs.speed * math.sin(obs.psi)
        b = -2.0 * (dpx * vtx + dpy * vty)

        h = d2 - r_safe * r_safe

        # Closing-speed + proximity tightening
        own_vx = state.u * math.cos(state.psi)
        own_vy = state.u * math.sin(state.psi)
        dvx = own_vx - vtx
        dvy = own_vy - vty
        dp_hat_x = dpx / d
        dp_hat_y = dpy / d
        v_close = max(0.0, -(dp_hat_x * dvx + dp_hat_y * dvy))
        own_speed = math.sqrt(own_vx**2 + own_vy**2)
        proximity = own_speed * max(0.0, 1.0 - d / (3.0 * r_safe))
        tightening = GAMMA_BACKUP * (v_close * d + proximity * r_safe)

        c_val = -ALPHA_BACKUP * h - b + tightening
        return a, c_val, h

    # =================================================================
    # Encounter-based direction selection (switching logic)
    # =================================================================

    def _select_direction(
        self, obs_label: str, enc_type: str, state: VesselState,
        obs: ObstacleState,
    ) -> str:
        """
        Select port or starboard avoidance based on COLREG rules.
        Includes hysteresis to prevent rapid switching.
        """
        preferred = COLREG_TURN_DIR.get(enc_type, "stbd")

        # Hysteresis: if we already committed to a direction for this
        # obstacle, keep it unless encounter type changes
        prev = self._active_dir.get(obs_label)
        if prev is not None and prev == preferred:
            return prev

        # Check both barriers — prefer the COLREG direction, but if
        # that barrier is deeply violated (h << 0), fall back
        h_pref = self.h_tc(
            state.x, state.y, state.psi,
            obs.x, obs.y, obs.length, obs.width,
            R_TURN, preferred,
        )
        alt = "port" if preferred == "stbd" else "stbd"
        h_alt = self.h_tc(
            state.x, state.y, state.psi,
            obs.x, obs.y, obs.length, obs.width,
            R_TURN, alt,
        )

        # Use preferred direction unless it's deeply violated and the
        # alternative is significantly better
        if h_pref < -R_SAFE_BASE * R_SAFE_BASE and h_alt > h_pref + R_SAFE_BASE * R_SAFE_BASE:
            direction = alt
        else:
            direction = preferred

        self._active_dir[obs_label] = direction
        return direction

    # =================================================================
    # Analytical QP solver (same as CC-CBF for fair comparison)
    # =================================================================

    @staticmethod
    def _solve_cbf_qp(
        u_nom: np.ndarray,
        A: np.ndarray,
        c_vec: np.ndarray,
        v_max: float,
    ) -> np.ndarray:
        """
        Solve:  min ||u - u_nom||^2
                s.t.  A[i] @ u >= c[i]   for all i
                      ||u|| <= v_max

        Iterative projection (same as CC-CBF Algorithm 1).
        """
        u = u_nom.copy()
        n = A.shape[0] if A.ndim == 2 else 0

        if n == 0:
            norm_val = np.linalg.norm(u)
            if norm_val > v_max:
                u = u * (v_max / norm_val)
            return u

        for _iteration in range(20):
            violated = False
            for i in range(n):
                ai = A[i]
                ci = c_vec[i]
                margin = ai @ u - ci
                if margin < 0:
                    a_norm_sq = ai @ ai
                    if a_norm_sq > 1e-12:
                        u = u + (-margin / a_norm_sq) * ai
                    violated = True
            if not violated:
                break

        norm_val = np.linalg.norm(u)
        if norm_val > v_max:
            u = u * (v_max / norm_val)

        return u

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

        self.llm_queried_this_step = False

        # Nominal: head toward goal at cruise speed
        nom_heading = self.nominal_heading(state, goal_x, goal_y)
        nom_speed = cfg.MAX_SPEED_MPS * 0.75

        if not obstacles:
            self._risk = 0.0
            self._man = "hold_course"
            return ControlCommand(
                desired_heading=nom_heading,
                desired_speed=nom_speed,
                manoeuvre="hold_course",
                explanation="TC-CBF: no obstacles",
            )

        # Build CBF constraints for nearby obstacles
        A_rows = []
        c_rows = []
        h_min = float("inf")
        closest_gw_enc = None
        closest_gw_dist = float("inf")

        for obs, enc in zip(obstacles, encounters):
            d = math.hypot(state.x - obs.x, state.y - obs.y)

            # Compute backup distance barrier (always active — safety net)
            a_d, c_d, h_d = self._distance_cbf_constraint(state, obs)

            # For COLREG encounters, also add the turning-circle barrier
            # to enforce the correct avoidance direction.
            # For static/unclassified obstacles, the distance barrier alone
            # suffices (turning-circle adds no COLREG value and can conflict).
            h_tc = float("inf")
            if enc.encounter_type in ("head_on", "crossing_give_way",
                                      "crossing_stand_on", "overtaking"):
                direction = self._select_direction(
                    obs.label, enc.encounter_type, state, obs
                )
                a_tc, c_tc, h_tc = self._cbf_constraint(
                    state, obs, enc, direction
                )

            h = min(h_tc, h_d)
            self._h_vals[obs.label] = h
            if h < h_min:
                h_min = h

            if d <= ACTIVATION_RANGE:
                A_rows.append(a_d)
                c_rows.append(c_d)
                if h_tc < float("inf"):
                    A_rows.append(a_tc)
                    c_rows.append(c_tc)

            # Track closest give-way encounter
            if enc.encounter_type in ("head_on", "crossing_give_way", "overtaking"):
                if d < closest_gw_dist:
                    closest_gw_dist = d
                    closest_gw_enc = enc

        # Risk
        self._risk = min(1.0, math.exp(-h_min / max(R_SAFE_BASE**2 * 0.5, 1.0)))

        if not A_rows:
            self._man = "hold_course"
            return ControlCommand(
                desired_heading=nom_heading,
                desired_speed=nom_speed,
                manoeuvre="hold_course",
                explanation="TC-CBF: safe (h={:.0f})".format(h_min),
            )

        # ---- COLREG directional bias on nominal ----
        biased_heading = nom_heading
        if closest_gw_enc is not None and closest_gw_dist < ACTIVATION_RANGE:
            proximity_factor = max(0.0, 1.0 - closest_gw_dist / ACTIVATION_RANGE)
            bias_deg = BIAS_DEG.get(closest_gw_enc.encounter_type, 25.0)
            bias_angle = -math.radians(bias_deg) * proximity_factor
            biased_heading = wrap_angle(nom_heading + bias_angle)

        u_nom = np.array([nom_speed * math.cos(biased_heading),
                          nom_speed * math.sin(biased_heading)])

        A = np.array(A_rows)
        c_vec = np.array(c_rows)

        # Solve analytical QP
        u_safe = self._solve_cbf_qp(u_nom, A, c_vec, cfg.MAX_SPEED_MPS)

        # Convert velocity → heading + speed
        des_heading = math.atan2(u_safe[1], u_safe[0])
        des_speed = float(np.linalg.norm(u_safe))
        des_speed = max(0.0, min(des_speed, cfg.MAX_SPEED_MPS))

        # Manoeuvre label
        heading_delta = wrap_angle(des_heading - nom_heading)
        speed_ratio = des_speed / max(nom_speed, 0.01)
        man = _label(heading_delta, speed_ratio)
        self._man = man

        if man == "hold_course":
            expl = "TC-CBF: safe (h={:.0f})".format(h_min)
        else:
            expl = "TC-CBF: {} (h={:.0f}, r={:.2f})".format(man, h_min, self._risk)

        return ControlCommand(
            desired_heading=des_heading,
            desired_speed=des_speed,
            manoeuvre=man,
            explanation=expl,
        )

    # =================================================================
    # Interface
    # =================================================================

    def get_risk_vector(self) -> Optional[RiskVector]:
        return RiskVector(Rg=self._risk)

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
        self._active_dir.clear()


# =====================================================================
# Manoeuvre labelling (same vocabulary as CC-CBF / metrics.py)
# =====================================================================

def _label(heading_delta: float, speed_ratio: float) -> str:
    stbd     = heading_delta < -math.radians(3)
    big_stbd = heading_delta < -math.radians(12)
    port     = heading_delta >  math.radians(3)
    slow     = speed_ratio   <  0.60

    if abs(heading_delta) < math.radians(3) and speed_ratio > 0.90:
        return "hold_course"
    if big_stbd:
        return "early_stbd"
    if stbd and slow:
        return "early_stbd"
    if stbd:
        return "late_stbd"
    if slow:
        return "slow_down"
    if port:
        return "emergency_port"
    return "early_stbd"
