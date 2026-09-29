"""
Data classes shared by the simulator, the controllers and the metrics.

Conventions: world frame x east / y north, headings in radians measured
counter-clockwise from +x; relative bearings are positive to port and
negative to starboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import data.config as cfg


@dataclass
class USVParameters:
    """
    Parameters of the own-ship USV (MBZIRC model, paper Table I).

    Fields marked [MBZIRC] are from the SDF; [APPROX] fields are
    approximations for this simulator.
    """
    # [MBZIRC] Hull geometry
    length: float = cfg.USV_LENGTH           # m
    width: float = cfg.USV_WIDTH             # m

    # [MBZIRC] Hydrodynamic drag
    x_u: float = cfg.X_U                     # N/(m/s)
    x_uu: float = cfg.X_UU                   # N/(m/s)^2

    # [MBZIRC] Speed / thrust limits
    max_speed: float = cfg.MAX_SPEED_MPS               # m/s
    max_total_thrust: float = cfg.MAX_TOTAL_THRUST     # N
    max_thruster_thrust: float = cfg.MAX_THRUSTER_THRUST  # N

    # [APPROX] Mass and inertia
    mass: float = cfg.USV_MASS                         # kg
    yaw_inertia: float = cfg.USV_YAW_INERTIA           # kg*m^2
    yaw_damping: float = cfg.USV_YAW_DAMPING           # N*m*s/rad
    thruster_lever_arm: float = cfg.THRUSTER_LEVER_ARM  # m


@dataclass
class VesselState:
    """State of the own-ship USV."""
    x: float = 0.0          # m, world x
    y: float = 0.0          # m, world y
    psi: float = 0.0        # rad, heading (0 = east, pi/2 = north)
    u: float = 0.0          # m/s, surge speed
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
    """State of a target vessel or static obstacle (constant velocity)."""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0        # rad, heading
    speed: float = 0.0      # m/s, forward speed
    length: float = cfg.DEFAULT_TARGET_LENGTH  # m
    width: float = cfg.DEFAULT_TARGET_WIDTH    # m
    is_static: bool = False
    label: str = "T1"       # unique per scenario; used as the target identity
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
class EncounterInfo:
    """COLREG classification of the encounter with one target."""
    # head_on, crossing_give_way, crossing_stand_on, overtaking, none
    encounter_type: str = "none"
    relative_bearing: float = 0.0  # rad, bearing from own ship to target
    relative_speed: float = 0.0    # m/s, magnitude of the relative velocity
    distance: float = float('inf')  # m, centre-to-centre
    dcpa: float = float('inf')     # m, distance at closest point of approach
    tcpa: float = float('inf')     # s, time to CPA
    target_label: str = ""
    target_side: str = "ahead"     # port, starboard, ahead


@dataclass
class ControlCommand:
    """High-level command produced by a controller for the autopilot."""
    desired_heading: float = 0.0   # rad
    desired_speed: float = 0.0     # m/s
    # hold_course, early_stbd, late_stbd, slow_down, stop, emergency_port
    manoeuvre: str = "hold_course"
    explanation: str = ""


@dataclass
class ThrusterCommand:
    """Low-level twin-thruster command."""
    T_L: float = 0.0  # N, left thruster
    T_R: float = 0.0  # N, right thruster


@dataclass
class StepRecord:
    """Everything logged for one simulation step."""
    time: float = 0.0
    state: Optional[VesselState] = None                # own ship after the step
    obstacles: List[ObstacleState] = field(default_factory=list)
    scalar_risk: float = 0.0                           # controller risk indicator
    encounter: Optional[EncounterInfo] = None          # closest detected target
    encounters: List[EncounterInfo] = field(default_factory=list)  # all detected targets
    command: Optional[ControlCommand] = None
    thruster: Optional[ThrusterCommand] = None
    min_distance: float = float('inf')                 # m, hull-to-hull clearance
    collision: bool = False


@dataclass
class SimulationMetrics:
    """Aggregated metrics of one completed run (paper Sec. VI-B)."""
    algorithm: str = ""
    scenario: str = ""
    collision_flag: bool = False
    min_separation: float = float('inf')   # m, minimum hull-to-hull clearance
    colreg_compliance: float = 1.0         # per-step rule adherence (COL)
    manoeuvre_commit_time: float = float('inf')
    path_length: float = 0.0
    nominal_distance: float = 0.0
    path_efficiency: float = 1.0           # nominal distance / path length
    avg_speed: float = 0.0
    peak_yaw_rate: float = 0.0
    total_thrust_integral: float = 0.0
    total_yaw_activity: float = 0.0
    peak_diff_thrust: float = 0.0
    sim_duration: float = 0.0


@dataclass
class ScenarioConfig:
    """Initial conditions and goal of a simulation scenario."""
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
