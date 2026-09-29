"""
Abstract base class of the collision-avoidance controllers.

Every controller implements :meth:`BaseController.compute_command` and
returns a :class:`~core.entities.ControlCommand` (desired heading and speed);
the shared autopilot converts it to thruster commands, so all controllers
act through identical low-level dynamics.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from core.entities import (
    VesselState, ObstacleState, ControlCommand, EncounterInfo, ScenarioConfig
)


class BaseController(ABC):
    """Abstract base for all collision-avoidance controllers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
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
        """
        Compute the high-level control command for one control period.

        Args:
            state: current own-ship state
            obstacles: obstacles reported by the LiDAR (within range)
            encounters: memoryless COLREG classification of each obstacle
            goal_x, goal_y: navigation goal
            time: current simulation time (s)
            dt: control period (s)
            scenario_config: optional scenario context

        Returns:
            ControlCommand with desired heading, speed and manoeuvre label
        """

    def get_scalar_risk(self) -> float:
        """Controller-specific scalar risk indicator (logged, not scored)."""
        return 0.0

    def reset(self) -> None:
        """Reset the controller's internal state before a new run."""

    @staticmethod
    def nominal_heading(state: VesselState, goal_x: float, goal_y: float) -> float:
        """Heading from the current position to the goal."""
        return math.atan2(goal_y - state.y, goal_x - state.x)


# ---------------------------------------------------------------------------
# Helpers shared by the velocity-space CBF controllers
# ---------------------------------------------------------------------------

def manoeuvre_label(heading_delta: float, speed_ratio: float) -> str:
    """
    Map a commanded heading change and speed ratio to a manoeuvre label.

    *heading_delta* is the commanded heading minus the goal heading
    (negative = starboard); *speed_ratio* is commanded / cruise speed.
    """
    stbd = heading_delta < -math.radians(3)
    big_stbd = heading_delta < -math.radians(12)
    port = heading_delta > math.radians(3)
    slow = speed_ratio < 0.60

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


def cyclic_projection(u_nom: np.ndarray, A: np.ndarray, c_vec: np.ndarray,
                      v_max: float, max_iter: int) -> np.ndarray:
    """
    Approximate solution of  min ||u - u_nom||^2  s.t.  A u >= c, ||u|| <= v_max
    by cyclic projection onto the violated half-planes (at most *max_iter*
    sweeps), followed by clipping to the speed limit.

    Used by the C3BF and TC-CBF baselines; CC-CBF solves the same planar
    problem exactly (see ``algorithms.cc_cbf``).
    """
    u = u_nom.copy()
    n = A.shape[0] if A.ndim == 2 else 0

    if n == 0:
        norm = np.linalg.norm(u)
        if norm > v_max:
            u = u * (v_max / norm)
        return u

    for _ in range(max_iter):
        violated = False
        for i in range(n):
            ai = A[i]
            margin = ai @ u - c_vec[i]
            if margin < 0:
                a_norm_sq = ai @ ai
                if a_norm_sq > 1e-12:
                    u = u + (-margin / a_norm_sq) * ai
                violated = True
        if not violated:
            break

    norm = np.linalg.norm(u)
    if norm > v_max:
        u = u * (v_max / norm)
    return u
