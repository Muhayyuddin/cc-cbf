"""
Data classes for the LC-CRI collision avoidance simulator.

All simulation entities, state vectors, commands, and metrics are defined here
using Python dataclasses with full type annotations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math
import numpy as np

import data.config as cfg


@dataclass
class USVParameters:
    """
    Parameters for the ownship USV, based on MBZIRC model.

    Fields marked [MBZIRC] are from the SDF.
    Fields marked [APPROX] are approximated for this simulator.
    """
    # [MBZIRC] Hull geometry
    length: float = cfg.USV_LENGTH           # m
    width: float = cfg.USV_WIDTH             # m
    body_height: float = cfg.USV_BODY_HEIGHT # m

    # [MBZIRC] Hydrodynamic drag
    x_u: float = cfg.X_U                    # N/(m/s)
    x_uu: float = cfg.X_UU                  # N/(m/s)^2

    # [MBZIRC] Speed / thrust limits
    max_speed: float = cfg.MAX_SPEED_MPS     # m/s
    max_total_thrust: float = cfg.MAX_TOTAL_THRUST   # N
    max_thruster_thrust: float = cfg.MAX_THRUSTER_THRUST  # N

    # [APPROX] Mass and inertia
    mass: float = cfg.USV_MASS               # kg
    yaw_inertia: float = cfg.USV_YAW_INERTIA # kg·m²
    yaw_damping: float = cfg.USV_YAW_DAMPING # N·m·s/rad
    thruster_lever_arm: float = cfg.THRUSTER_LEVER_ARM  # m


@dataclass
class VesselState:
    """State of the ownship USV."""
    x: float = 0.0          # m, world x
    y: float = 0.0          # m, world y
    psi: float = 0.0        # rad, heading (0 = east, pi/2 = north)
    u: float = 0.0          # m/s, forward speed (surge)
    r: float = 0.0          # rad/s, yaw rate

    def copy(self) -> VesselState:
        return VesselState(self.x, self.y, self.psi, self.u, self.r)

    @property
    def vx(self) -> float:
        """World-frame x velocity."""
        return self.u * math.cos(self.psi)

    @property
    def vy(self) -> float:
        """World-frame y velocity."""
        return self.u * math.sin(self.psi)


@dataclass
class ObstacleState:
    """State of a target vessel or static obstacle."""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0        # heading
    speed: float = 0.0      # m/s, forward speed
    length: float = cfg.DEFAULT_TARGET_LENGTH  # m
    width: float = cfg.DEFAULT_TARGET_WIDTH    # m
    is_static: bool = False
    label: str = "T1"
    vessel_type: str = "power_driven"  # power_driven, sailing, restricted

    def copy(self) -> ObstacleState:
        return ObstacleState(
            self.x, self.y, self.psi, self.speed,
            self.length, self.width, self.is_static,
            self.label, self.vessel_type
        )

    @property
    def vx(self) -> float:
        return self.speed * math.cos(self.psi)

    @property
    def vy(self) -> float:
        return self.speed * math.sin(self.psi)


@dataclass
class RiskVector:
    """Five-component risk vector for LC-CRI."""
    Rg: float = 0.0  # geometric risk
    Rd: float = 0.0  # dynamic risk
    Rr: float = 0.0  # regulatory risk
    Rs: float = 0.0  # semantic risk
    Ri: float = 0.0  # intent risk

    def to_array(self) -> np.ndarray:
        return np.array([self.Rg, self.Rd, self.Rr, self.Rs, self.Ri])

    @staticmethod
    def from_array(arr: np.ndarray) -> RiskVector:
        return RiskVector(Rg=arr[0], Rd=arr[1], Rr=arr[2], Rs=arr[3], Ri=arr[4])

    @property
    def weighted_sum(self) -> float:
        w = cfg.LCRI_RISK_WEIGHTS
        return (w['Rg'] * self.Rg + w['Rd'] * self.Rd + w['Rr'] * self.Rr +
                w['Rs'] * self.Rs + w['Ri'] * self.Ri)

    def divergence_from(self, other: RiskVector) -> float:
        """Euclidean distance between two risk vectors."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))


@dataclass
class EncounterInfo:
    """Classification of an encounter with a target."""
    encounter_type: str = "none"  # head_on, crossing_give_way, crossing_stand_on, overtaking, none
    relative_bearing: float = 0.0  # rad, bearing from ownship to target
    relative_speed: float = 0.0    # m/s
    distance: float = float('inf')
    dcpa: float = float('inf')     # m, distance at closest point of approach
    tcpa: float = float('inf')     # s, time to CPA
    target_label: str = ""
    target_side: str = "ahead"     # port, starboard, ahead, astern


@dataclass
class ControlCommand:
    """High-level control command from an algorithm."""
    desired_heading: float = 0.0   # rad
    desired_speed: float = 0.0     # m/s
    manoeuvre: str = "hold_course"  # hold_course, early_stbd, late_stbd, slow_down, stop, emergency_port
    explanation: str = ""


@dataclass
class ThrusterCommand:
    """Low-level twin-thruster command."""
    T_L: float = 0.0  # N, left thruster
    T_R: float = 0.0  # N, right thruster


@dataclass
class PseudoLLMResponse:
    """Response from the pseudo-LLM module."""
    distribution: Dict[str, float] = field(default_factory=lambda: {
        'hold_course': 1.0, 'early_stbd': 0.0, 'late_stbd': 0.0,
        'slow_down': 0.0, 'stop': 0.0, 'emergency_port': 0.0
    })
    recommended_delta_heading: float = 0.0  # rad
    recommended_speed_factor: float = 1.0   # 0..1
    explanation: str = "No action needed."


@dataclass
class StepRecord:
    """Record of a single simulation step for logging."""
    time: float = 0.0
    state: Optional[VesselState] = None
    obstacles: List[ObstacleState] = field(default_factory=list)
    risk: Optional[RiskVector] = None
    scalar_risk: float = 0.0
    encounter: Optional[EncounterInfo] = None          # closest obstacle's encounter
    encounters: List['EncounterInfo'] = field(default_factory=list)  # all per-obstacle encounters
    command: Optional[ControlCommand] = None
    thruster: Optional[ThrusterCommand] = None
    min_distance: float = float('inf')
    llm_queried: bool = False
    collision: bool = False


@dataclass
class SimulationMetrics:
    """Aggregated metrics for a completed simulation run."""
    algorithm: str = ""
    scenario: str = ""
    collision_flag: bool = False
    min_separation: float = float('inf')
    colreg_compliance: float = 1.0
    manoeuvre_commit_time: float = float('inf')
    path_length: float = 0.0
    nominal_distance: float = 0.0
    path_efficiency: float = 1.0
    total_llm_queries: int = 0
    avg_speed: float = 0.0
    peak_yaw_rate: float = 0.0
    total_thrust_integral: float = 0.0
    total_yaw_activity: float = 0.0
    peak_diff_thrust: float = 0.0
    sim_duration: float = 0.0


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    name: str = "head_on"
    ownship_start: VesselState = field(default_factory=VesselState)
    ownship_goal_x: float = 200.0
    ownship_goal_y: float = 0.0
    targets: List[ObstacleState] = field(default_factory=list)
    context_flags: Dict[str, bool] = field(default_factory=lambda: {
        'narrow_channel': False,
        'reduced_visibility': False,
    })
    description: str = ""
