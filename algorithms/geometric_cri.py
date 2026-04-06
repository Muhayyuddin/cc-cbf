"""
Baseline 1: Geometric CRI controller.

Uses scalar CRI based on DCPA / TCPA and distance.
When CRI exceeds threshold, performs starboard turn and/or slow down.
"""

import math
from typing import List, Optional
from algorithms.base_controller import BaseController
from core.entities import (
    VesselState, ObstacleState, ControlCommand, EncounterInfo,
    RiskVector, ScenarioConfig
)
from core.geometry import wrap_angle, compute_dcpa_tcpa, center_distance
from algorithms.rule_based_colreg import _IsotropicCBFFilter
import data.config as cfg


class GeometricCRIController(BaseController):
    """
    Geometric CRI baseline.

    Computes a scalar collision risk index from distance, DCPA, and TCPA.
    Triggers avoidance when CRI > threshold.
    """

    def __init__(self):
        super().__init__(name="Geometric CRI")
        self.safety_filter = _IsotropicCBFFilter()
        self._scalar_risk = 0.0

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

        # Nominal heading to goal
        nominal_heading = self.nominal_heading(state, goal_x, goal_y)
        nominal_speed = cfg.MAX_SPEED_MPS * 0.75  # cruise at 75% max

        # Compute scalar CRI for all obstacles
        max_cri = 0.0
        worst_enc = None
        for enc in encounters:
            cri = self._compute_cri(enc)
            if cri > max_cri:
                max_cri = cri
                worst_enc = enc

        self._scalar_risk = max_cri

        # Decision logic
        if max_cri > cfg.GEO_CRI_THRESHOLD and worst_enc is not None:
            # Avoidance: starboard turn + slow down
            avoidance_heading = wrap_angle(nominal_heading - cfg.GEO_CRI_STBD_TURN)
            avoidance_speed = nominal_speed * cfg.GEO_CRI_SLOW_FACTOR

            # Scale by urgency
            urgency = min(1.0, max_cri / 1.0)
            desired_heading = wrap_angle(
                nominal_heading * (1 - urgency) + avoidance_heading * urgency
            )
            desired_speed = nominal_speed * (1 - urgency) + avoidance_speed * urgency

            manoeuvre = "early_stbd" if urgency > 0.5 else "late_stbd"
            if desired_speed < nominal_speed * 0.4:
                manoeuvre = "slow_down"

            cmd = ControlCommand(
                desired_heading=desired_heading,
                desired_speed=max(0.0, desired_speed),
                manoeuvre=manoeuvre,
                explanation=f"CRI={max_cri:.2f} > threshold, avoiding"
            )
        else:
            cmd = ControlCommand(
                desired_heading=nominal_heading,
                desired_speed=nominal_speed,
                manoeuvre="hold_course",
                explanation=f"CRI={max_cri:.2f}, clear"
            )

        # Safety filter
        cmd = self.safety_filter.filter_command(state, cmd, obstacles)
        return cmd

    def _compute_cri(self, enc: EncounterInfo) -> float:
        """
        Compute scalar collision risk index.

        CRI = w_d * R_distance + w_dcpa * R_dcpa + w_tcpa * R_tcpa
        """
        # Distance risk
        if enc.distance < cfg.D_SAFE:
            r_dist = 1.0
        elif enc.distance > cfg.AVOIDANCE_DOMAIN * 2:
            r_dist = 0.0
        else:
            r_dist = 1.0 - (enc.distance - cfg.D_SAFE) / (cfg.AVOIDANCE_DOMAIN * 2 - cfg.D_SAFE)

        # DCPA risk
        if enc.dcpa < cfg.D_SAFE:
            r_dcpa = 1.0 - enc.dcpa / cfg.D_SAFE
        else:
            r_dcpa = 0.0

        # TCPA risk
        if enc.tcpa <= 0 or enc.tcpa > 60:
            r_tcpa = 0.0
        elif enc.tcpa < 10:
            r_tcpa = 1.0 - enc.tcpa / 10.0
        else:
            r_tcpa = max(0.0, 0.5 - enc.tcpa / 120.0)

        cri = 0.4 * r_dist + 0.35 * r_dcpa + 0.25 * r_tcpa
        return min(1.0, cri)

    def get_scalar_risk(self) -> float:
        return self._scalar_risk

    def reset(self):
        super().reset()
        self._scalar_risk = 0.0
