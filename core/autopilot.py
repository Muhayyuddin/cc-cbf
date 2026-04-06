"""
Shared low-level autopilot / thruster controller for the MBZIRC USV.

All high-level algorithms output (desired_heading, desired_speed).
This module converts those into twin-thruster commands (T_L, T_R)
through a shared speed + heading controller with feedforward.
"""

import math
import numpy as np
from core.entities import (
    USVParameters, VesselState, ControlCommand, ThrusterCommand
)
from core.geometry import wrap_angle
import data.config as cfg


class Autopilot:
    """
    Low-level controller converting desired heading/speed to thruster commands.

    Speed controller: PI with drag feedforward.
    Heading controller: PD on heading error with yaw rate damping.
    Thruster allocation: common-mode + differential thrust.
    """

    def __init__(self, params: USVParameters = None):
        self.params = params or USVParameters()
        self._speed_integral = 0.0

        # Gains
        self.kp_speed = cfg.KP_SPEED
        self.ki_speed = cfg.KI_SPEED
        self.kp_heading = cfg.KP_HEADING
        self.kd_heading = cfg.KD_HEADING
        self.k_yaw_thrust = cfg.K_YAW_TO_THRUST

    def reset(self):
        """Reset integrator state."""
        self._speed_integral = 0.0

    def compute_thrust(self, state: VesselState, cmd: ControlCommand,
                       dt: float) -> ThrusterCommand:
        """
        Compute left/right thruster commands from high-level control command.

        Args:
            state: current USV state
            cmd: desired heading and speed from algorithm
            dt: time step

        Returns:
            ThrusterCommand with T_L and T_R
        """
        p = self.params

        # --- Speed controller with drag feedforward ---
        speed_error = cmd.desired_speed - state.u
        self._speed_integral += speed_error * dt
        # Anti-windup: clamp integral
        self._speed_integral = np.clip(self._speed_integral, -50.0, 50.0)

        # Feedforward: thrust to overcome drag at desired speed
        u_ref = cmd.desired_speed
        feedforward = p.x_u * u_ref + p.x_uu * abs(u_ref) * u_ref

        T_common = (feedforward +
                     self.kp_speed * speed_error +
                     self.ki_speed * self._speed_integral)

        # --- Heading controller ---
        heading_error = wrap_angle(cmd.desired_heading - state.psi)
        yaw_cmd = self.kp_heading * heading_error - self.kd_heading * state.r

        # Differential thrust for yaw
        T_diff = self.k_yaw_thrust * yaw_cmd

        # --- Thruster allocation ---
        # T_L = T_common - T_diff  (positive T_diff => more right thrust => turn port)
        # T_R = T_common + T_diff
        T_L = T_common - T_diff
        T_R = T_common + T_diff

        # Clamp to thruster limits
        # Allow small reverse for braking, but primarily forward thrust
        T_min = -p.max_thruster_thrust * 0.1  # 10% reverse
        T_max = p.max_thruster_thrust

        T_L = float(np.clip(T_L, T_min, T_max))
        T_R = float(np.clip(T_R, T_min, T_max))

        return ThrusterCommand(T_L=T_L, T_R=T_R)
