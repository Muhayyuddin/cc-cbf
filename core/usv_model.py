"""
MBZIRC-based USV dynamics model.

Implements a simplified 3-DOF planar dynamics model for the ownship,
using parameters from the MBZIRC USV SDF model where available.

State: [x, y, psi, u, r]
Inputs: [T_L, T_R] (left and right thruster forces)

Each thruster is mounted at the longitudinal midpoint of its pontoon
(SDF pose x=0), offset laterally by ell = 1.348 m from the centreline.

Dynamics:
    m  * du/dt  = T_L + T_R - x_u * u - x_uu * |u| * u
    Iz * dr/dt  = ell * (T_R - T_L) - N_r * r
    dx/dt   = u * cos(psi)
    dy/dt   = u * sin(psi)
    dpsi/dt = r
"""

import math
import numpy as np
from core.entities import USVParameters, VesselState, ThrusterCommand
import data.config as cfg


class USVModel:
    """
    Simplified planar dynamics model for the MBZIRC USV.

    Uses Euler integration for state updates.
    """

    def __init__(self, params: USVParameters = None):
        self.params = params or USVParameters()

    def step(self, state: VesselState, cmd: ThrusterCommand, dt: float) -> VesselState:
        """
        Advance the USV state by one time step.

        Args:
            state: current vessel state
            cmd: thruster command (T_L, T_R)
            dt: time step in seconds

        Returns:
            new VesselState after integration
        """
        p = self.params
        T_L = np.clip(cmd.T_L, -p.max_thruster_thrust * 0.1, p.max_thruster_thrust)
        T_R = np.clip(cmd.T_R, -p.max_thruster_thrust * 0.1, p.max_thruster_thrust)

        # Total and differential thrust
        T_total = T_L + T_R
        T_diff = T_R - T_L  # positive => turn to port (left), negative => turn to starboard

        # Surge dynamics: m * du/dt = T_total - x_u * u - x_uu * |u| * u
        drag = p.x_u * state.u + p.x_uu * abs(state.u) * state.u
        du_dt = (T_total - drag) / p.mass

        # Yaw dynamics: Iz * dr/dt = ell * (T_R - T_L) - N_r * r
        # ell = lateral thruster offset = 1.348 m (thrusters at mid-hull)
        dr_dt = (p.thruster_lever_arm * T_diff - p.yaw_damping * state.r) / p.yaw_inertia

        # Integrate
        new_u = state.u + du_dt * dt
        new_r = state.r + dr_dt * dt

        # Clamp speed
        new_u = np.clip(new_u, -0.5, p.max_speed)  # allow small reverse

        # Clamp yaw rate to reasonable bounds
        max_yaw_rate = 1.0  # rad/s, physical limit for 6m USV
        new_r = np.clip(new_r, -max_yaw_rate, max_yaw_rate)

        # Kinematics
        new_psi = state.psi + state.r * dt  # use current r for semi-implicit
        new_x = state.x + state.u * math.cos(state.psi) * dt
        new_y = state.y + state.u * math.sin(state.psi) * dt

        # Wrap heading
        while new_psi > math.pi:
            new_psi -= 2.0 * math.pi
        while new_psi < -math.pi:
            new_psi += 2.0 * math.pi

        return VesselState(x=new_x, y=new_y, psi=new_psi, u=float(new_u), r=float(new_r))

    def steady_state_thrust(self, speed: float) -> float:
        """
        Compute total thrust needed to maintain given forward speed.

        From: T = x_u * u + x_uu * |u| * u
        """
        p = self.params
        return p.x_u * speed + p.x_uu * abs(speed) * speed
