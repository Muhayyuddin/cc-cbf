"""
Predictive safety filter shared by the Rule-COLREG and Geo-CRI baselines.

A CBF-inspired filter on the candidate (desired_heading, desired_speed):
a CBF_HORIZON = 3 s, CBF_STEPS = 6 step rollout of the vessel model under
the candidate command; commands whose predicted separation falls below
CBF_MIN_SEPARATION are rejected and replaced by the nearest safe
heading / speed on a fixed grid (starboard preferred).  This is the filter
with which the paper results (Tables III and VII) were generated.
"""

import math
from typing import List, Optional

import data.config as cfg
from core.autopilot import Autopilot
from core.entities import VesselState, ObstacleState, ControlCommand, USVParameters
from core.geometry import center_distance, wrap_angle
from core.usv_model import USVModel


class CBFSafetyFilter:
    """
    Simplified CBF-style safety filter.

    Predicts short-horizon motion under candidate commands and rejects
    commands that would reduce separation below the safety threshold.
    Projects to nearest safe command.
    """

    def __init__(self, params: Optional[USVParameters] = None):
        self.params = params or USVParameters()
        self.usv_model = USVModel(self.params)
        # Note: the prediction autopilot's speed integrator persists across
        # rollouts (as in the implementation behind the paper results).
        self.autopilot = Autopilot(self.params)
        self.horizon = cfg.CBF_HORIZON
        self.n_steps = cfg.CBF_STEPS
        self.min_sep = cfg.CBF_MIN_SEPARATION

    def filter_command(
        self,
        state: VesselState,
        cmd: ControlCommand,
        obstacles: List[ObstacleState],
    ) -> ControlCommand:
        """
        Filter the proposed command through the safety layer.

        If the command is safe, return it unchanged.
        If unsafe, search for the nearest safe alternative.
        """
        if not obstacles:
            return cmd

        if self._is_safe(state, cmd, obstacles):
            return cmd

        return self._find_safe_command(state, cmd, obstacles)

    def _predict_trajectory(
        self,
        state: VesselState,
        cmd: ControlCommand,
    ) -> List[VesselState]:
        """Predict short-horizon trajectory under given command."""
        dt = self.horizon / self.n_steps
        trajectory = [state]
        current = state.copy()

        for _ in range(self.n_steps):
            thrust = self.autopilot.compute_thrust(current, cmd, dt)
            current = self.usv_model.step(current, thrust, dt)
            trajectory.append(current)
            current = current.copy()

        return trajectory

    def _predict_obstacle(self, obs: ObstacleState, dt: float, steps: int) -> List[ObstacleState]:
        """Predict obstacle trajectory (constant velocity)."""
        trajectory = [obs.copy()]
        for i in range(1, steps + 1):
            new_obs = obs.copy()
            new_obs.x = obs.x + obs.speed * math.cos(obs.psi) * dt * i
            new_obs.y = obs.y + obs.speed * math.sin(obs.psi) * dt * i
            trajectory.append(new_obs)
        return trajectory

    def _min_predicted_separation(
        self,
        state: VesselState,
        cmd: ControlCommand,
        obstacles: List[ObstacleState],
    ) -> float:
        """Compute minimum predicted separation over horizon."""
        dt = self.horizon / self.n_steps
        own_traj = self._predict_trajectory(state, cmd)

        half_diag_own = math.sqrt(self.params.length**2 + self.params.width**2) / 2.0

        min_sep = float('inf')
        for obs in obstacles:
            half_diag_obs = math.sqrt(obs.length**2 + obs.width**2) / 2.0
            buffer = half_diag_own + half_diag_obs
            obs_traj = self._predict_obstacle(obs, dt, self.n_steps)
            for own_state, obs_state in zip(own_traj, obs_traj):
                d = center_distance(own_state.x, own_state.y,
                                     obs_state.x, obs_state.y) - buffer
                if d < min_sep:
                    min_sep = d

        return max(0.0, min_sep)

    def _is_safe(self, state: VesselState, cmd: ControlCommand,
                 obstacles: List[ObstacleState]) -> bool:
        """Check if command leads to safe predicted trajectory."""
        return self._min_predicted_separation(state, cmd, obstacles) >= self.min_sep

    def _find_safe_command(
        self,
        state: VesselState,
        cmd: ControlCommand,
        obstacles: List[ObstacleState],
    ) -> ControlCommand:
        """
        Search for nearest safe command to the proposed one.

        Strategy: try heading offsets (preferring starboard per COLREG)
        and speed reductions.
        """
        best_cmd = None
        best_cost = float('inf')

        heading_offsets = [0, -10, -20, -30, -45, -60, -80, 10, 20, 30]
        speed_factors = [1.0, 0.7, 0.4, 0.0]

        for h_deg in heading_offsets:
            for sf in speed_factors:
                h_rad = math.radians(h_deg)
                trial = ControlCommand(
                    desired_heading=wrap_angle(cmd.desired_heading + h_rad),
                    desired_speed=cmd.desired_speed * sf,
                    manoeuvre=cmd.manoeuvre,
                    explanation=cmd.explanation,
                )

                if self._is_safe(state, trial, obstacles):
                    cost = abs(h_rad) * 2.0 + (1.0 - sf) * 3.0
                    if h_rad > 0:
                        cost += 1.0

                    if cost < best_cost:
                        best_cost = cost
                        best_cmd = trial

        if best_cmd is not None:
            h_diff = wrap_angle(best_cmd.desired_heading - cmd.desired_heading)
            if abs(h_diff) > math.radians(5):
                if h_diff < 0:
                    best_cmd.manoeuvre = "early_stbd"
                else:
                    best_cmd.manoeuvre = "emergency_port"
            if best_cmd.desired_speed < cmd.desired_speed * 0.5:
                best_cmd.manoeuvre = "slow_down"
            best_cmd.explanation += " [CBF-filtered]"
            return best_cmd

        return ControlCommand(
            desired_heading=state.psi,
            desired_speed=0.0,
            manoeuvre="stop",
            explanation="CBF safety: no safe heading found, stopping. " + cmd.explanation,
        )
