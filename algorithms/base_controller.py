"""
Abstract base controller for collision avoidance algorithms.

All algorithms must implement compute_command() and return a ControlCommand
with desired heading and speed. The shared autopilot then converts this
to thruster commands.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from core.entities import (
    VesselState, ObstacleState, ControlCommand, EncounterInfo,
    RiskVector, ScenarioConfig
)


class BaseController(ABC):
    """Abstract base for all collision avoidance algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.llm_query_count = 0
        self.llm_queried_this_step = False

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
        Compute high-level control command.

        Args:
            state: current ownship state
            obstacles: list of current obstacle states
            encounters: classified encounters for each obstacle
            goal_x, goal_y: navigation goal
            time: current simulation time
            dt: time step
            scenario_config: optional scenario context

        Returns:
            ControlCommand with desired heading, speed, and manoeuvre label
        """
        ...

    def get_risk_vector(self) -> Optional[RiskVector]:
        """Return current risk vector if available (LC-CRI only)."""
        return None

    def get_scalar_risk(self) -> float:
        """Return scalar risk value."""
        return 0.0

    def reset(self):
        """Reset controller state."""
        self.llm_query_count = 0
        self.llm_queried_this_step = False

    @staticmethod
    def nominal_heading(state: VesselState, goal_x: float, goal_y: float) -> float:
        """Compute heading from current position to goal."""
        import math
        dx = goal_x - state.x
        dy = goal_y - state.y
        return math.atan2(dy, dx)
