"""
Baseline 2: Rule-based COLREG controller.

Deterministic encounter classification mapped to fixed manoeuvres:
- head_on -> starboard turn
- crossing_give_way -> early starboard or slow down
- crossing_stand_on -> hold course (emergency if needed)
- overtaking -> starboard passing
"""

import math
from typing import List, Optional
from algorithms.base_controller import BaseController
from core.entities import (
    VesselState, ObstacleState, ControlCommand, EncounterInfo,
    ScenarioConfig
)
from core.geometry import wrap_angle
import data.config as cfg


class _IsotropicCBFFilter:
    """Minimal isotropic CBF safety filter (replaces deleted cbf_filter stub)."""

    def filter_command(
        self,
        state: VesselState,
        cmd: ControlCommand,
        obstacles: List[ObstacleState],
    ) -> ControlCommand:
        if not obstacles:
            return cmd
        # Find closest obstacle
        closest = min(obstacles, key=lambda o: math.hypot(o.x - state.x, o.y - state.y))
        dist = math.hypot(closest.x - state.x, closest.y - state.y)
        r_safe = cfg.D_SAFE + 0.5 * (cfg.USV_LENGTH + closest.length)
        if dist < r_safe * 1.5:
            # Back off speed proportionally when inside 1.5 × safety radius
            factor = max(0.3, (dist - r_safe) / (r_safe * 0.5)) if dist > r_safe else 0.3
            cmd = ControlCommand(
                desired_heading=cmd.desired_heading,
                desired_speed=cmd.desired_speed * factor,
                manoeuvre=cmd.manoeuvre,
                explanation=cmd.explanation + " [CBF speed trim]",
            )
        return cmd


class RuleBasedCOLREGController(BaseController):
    """
    Deterministic COLREG rule-based controller.

    Maps encounter types directly to manoeuvre actions.
    """

    def __init__(self):
        super().__init__(name="Rule-based COLREG")
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

        nominal_heading = self.nominal_heading(state, goal_x, goal_y)
        nominal_speed = cfg.MAX_SPEED_MPS * 0.75

        # Find most urgent encounter
        worst_enc = None
        worst_urgency = 0.0
        for enc in encounters:
            urg = self._urgency(enc)
            if urg > worst_urgency:
                worst_urgency = urg
                worst_enc = enc

        self._scalar_risk = worst_urgency

        if worst_enc is None or worst_urgency < 0.1:
            cmd = ControlCommand(
                desired_heading=nominal_heading,
                desired_speed=nominal_speed,
                manoeuvre="hold_course",
                explanation="No encounter, proceeding to goal"
            )
            cmd = self.safety_filter.filter_command(state, cmd, obstacles)
            return cmd

        # Apply COLREG rules
        cmd = self._apply_rules(state, worst_enc, nominal_heading, nominal_speed, worst_urgency)
        cmd = self.safety_filter.filter_command(state, cmd, obstacles)
        return cmd

    def _apply_rules(self, state: VesselState, enc: EncounterInfo,
                     nominal_heading: float, nominal_speed: float,
                     urgency: float) -> ControlCommand:
        """Map encounter type to manoeuvre."""

        if enc.encounter_type == "head_on":
            # Rule 14: Both vessels turn starboard
            turn = math.radians(25) * min(1.0, urgency * 1.5)
            heading = wrap_angle(nominal_heading - turn)  # starboard
            speed = nominal_speed * (1.0 - 0.3 * urgency)
            return ControlCommand(
                desired_heading=heading,
                desired_speed=max(0.5, speed),
                manoeuvre="early_stbd",
                explanation=f"Rule 14 head-on: starboard turn {math.degrees(turn):.0f}°"
            )

        elif enc.encounter_type == "crossing_give_way":
            # Rule 15: Give-way vessel shall avoid crossing ahead
            turn = math.radians(30) * min(1.0, urgency * 1.5)
            heading = wrap_angle(nominal_heading - turn)
            speed = nominal_speed * (1.0 - 0.4 * urgency)
            manoeuvre = "early_stbd" if urgency > 0.3 else "slow_down"
            return ControlCommand(
                desired_heading=heading,
                desired_speed=max(0.3, speed),
                manoeuvre=manoeuvre,
                explanation=f"Rule 15 crossing give-way: yield to starboard"
            )

        elif enc.encounter_type == "crossing_stand_on":
            # Rule 17: Stand-on vessel maintains course
            if urgency > 0.8:
                # Emergency: target not giving way
                turn = math.radians(20)
                heading = wrap_angle(nominal_heading - turn)  # starboard as last resort
                return ControlCommand(
                    desired_heading=heading,
                    desired_speed=nominal_speed * 0.3,
                    manoeuvre="emergency_port",
                    explanation="Rule 17(b): stand-on vessel taking emergency action"
                )
            else:
                return ControlCommand(
                    desired_heading=nominal_heading,
                    desired_speed=nominal_speed,
                    manoeuvre="hold_course",
                    explanation="Rule 17: stand-on, maintaining course and speed"
                )

        elif enc.encounter_type == "overtaking":
            # Rule 13: Keep clear when overtaking
            turn = math.radians(20) * min(1.0, urgency * 1.5)
            heading = wrap_angle(nominal_heading - turn)
            speed = nominal_speed * (1.0 - 0.2 * urgency)
            return ControlCommand(
                desired_heading=heading,
                desired_speed=max(0.5, speed),
                manoeuvre="early_stbd",
                explanation="Rule 13 overtaking: pass to starboard"
            )

        else:
            # Static obstacles or unknown
            if urgency > 0.3:
                turn = math.radians(25) * urgency
                heading = wrap_angle(nominal_heading - turn)
                speed = nominal_speed * (1.0 - 0.3 * urgency)
                return ControlCommand(
                    desired_heading=heading,
                    desired_speed=max(0.5, speed),
                    manoeuvre="early_stbd",
                    explanation="Obstacle avoidance: starboard turn"
                )
            return ControlCommand(
                desired_heading=nominal_heading,
                desired_speed=nominal_speed,
                manoeuvre="hold_course",
                explanation="No action needed"
            )

    def _urgency(self, enc: EncounterInfo) -> float:
        """Simple urgency from distance and DCPA."""
        urgency = 0.0
        if enc.distance < cfg.AVOIDANCE_DOMAIN:
            urgency = max(urgency, 1.0 - enc.distance / cfg.AVOIDANCE_DOMAIN)
        if enc.dcpa < cfg.D_SAFE * 2:
            urgency = max(urgency, 1.0 - enc.dcpa / (cfg.D_SAFE * 2))
        if 0 < enc.tcpa < 20:
            urgency = max(urgency, 1.0 - enc.tcpa / 20.0)
        return min(1.0, urgency)

    def get_scalar_risk(self) -> float:
        return self._scalar_risk

    def reset(self):
        super().reset()
        self._scalar_risk = 0.0
