"""
Main simulation engine.

Runs a single scenario with a given algorithm, stepping the MBZIRC USV
dynamics and obstacle models, recording all data for analysis.
"""

import math
import copy
from typing import List, Optional, Dict
from core.entities import (
    USVParameters, VesselState, ObstacleState, ControlCommand,
    ThrusterCommand, StepRecord, SimulationMetrics, ScenarioConfig,
    RiskVector, EncounterInfo
)
from core.usv_model import USVModel
from core.autopilot import Autopilot
from core.colregs import classify_encounter
from core.geometry import center_distance, min_distance_rects, rect_overlap
from core.metrics import compute_metrics
from algorithms.base_controller import BaseController
import data.config as cfg


class Simulator:
    """
    Simulation engine for a single algorithm on a single scenario.

    Manages:
    - MBZIRC USV dynamics
    - Obstacle motion
    - Encounter classification
    - Algorithm control loop
    - Data recording
    """

    def __init__(self, scenario: ScenarioConfig, controller: BaseController,
                 params: USVParameters = None):
        self.scenario = scenario
        self.controller = controller
        self.params = params or USVParameters()
        self.usv_model = USVModel(self.params)
        self.autopilot = Autopilot(self.params)

        # State
        self.state = scenario.ownship_start.copy()
        self.obstacles = [t.copy() for t in scenario.targets]
        self.time = 0.0
        self.step_count = 0

        # Records
        self.records: List[StepRecord] = []
        self.finished = False

    def reset(self):
        """Reset simulation to initial conditions."""
        self.state = self.scenario.ownship_start.copy()
        self.obstacles = [t.copy() for t in self.scenario.targets]
        self.time = 0.0
        self.step_count = 0
        self.records = []
        self.finished = False
        self.controller.reset()
        self.autopilot.reset()

    def step(self, dt: float = None) -> StepRecord:
        """
        Advance simulation by one time step.

        Returns:
            StepRecord with all data for this step
        """
        if dt is None:
            dt = cfg.SIM_DT

        if self.finished:
            return self.records[-1] if self.records else StepRecord()

        # 1. Sensor gating: only obstacles within LiDAR detection range
        #    are "visible" to the controller (MBZIRC planar LiDAR model).
        #    Ground-truth distances are still used for safety metrics.
        detected_obstacles = []
        for obs in self.obstacles:
            d = center_distance(self.state.x, self.state.y, obs.x, obs.y)
            if d <= cfg.LIDAR_RANGE:
                detected_obstacles.append(obs)

        # 2. Classify encounters (only for detected obstacles)
        encounters = []
        for obs in detected_obstacles:
            enc = classify_encounter(self.state, obs)
            encounters.append(enc)

        # 3. Compute minimum distance (rectangle-aware, ALL obstacles for metrics)
        min_dist = float('inf')
        has_collision = False
        for obs in self.obstacles:
            d = min_distance_rects(
                self.state.x, self.state.y,
                self.params.length, self.params.width, self.state.psi,
                obs.x, obs.y, obs.length, obs.width, obs.psi
            )
            if d < min_dist:
                min_dist = d
            if d <= 0:
                has_collision = True

        # 4. Get control command from algorithm (only detected obstacles)
        cmd = self.controller.compute_command(
            state=self.state,
            obstacles=detected_obstacles,
            encounters=encounters,
            goal_x=self.scenario.ownship_goal_x,
            goal_y=self.scenario.ownship_goal_y,
            time=self.time,
            dt=dt,
            scenario_config=self.scenario,
        )

        # 5. Convert to thruster commands via shared autopilot
        thrust = self.autopilot.compute_thrust(self.state, cmd, dt)

        # 6. Step USV dynamics
        self.state = self.usv_model.step(self.state, thrust, dt)

        # 7. Step obstacles (constant velocity)
        for obs in self.obstacles:
            if not obs.is_static:
                obs.x += obs.speed * math.cos(obs.psi) * dt
                obs.y += obs.speed * math.sin(obs.psi) * dt

        # 8. Record
        risk_vec = self.controller.get_risk_vector()
        scalar_risk = self.controller.get_scalar_risk()

        # Get the closest encounter for the record
        closest_enc = None
        if encounters:
            closest_enc = min(encounters, key=lambda e: e.distance)

        record = StepRecord(
            time=self.time,
            state=self.state.copy(),
            obstacles=[o.copy() for o in self.obstacles],
            risk=risk_vec,
            scalar_risk=scalar_risk,
            encounter=closest_enc,
            encounters=list(encounters),
            command=cmd,
            thruster=thrust,
            min_distance=min_dist,
            llm_queried=self.controller.llm_queried_this_step,
            collision=has_collision,
        )
        self.records.append(record)

        # 9. Advance time
        self.time += dt
        self.step_count += 1

        # 10. Check termination
        dist_to_goal = center_distance(
            self.state.x, self.state.y,
            self.scenario.ownship_goal_x, self.scenario.ownship_goal_y
        )
        if dist_to_goal < 10.0:
            self.finished = True
        if self.time > cfg.SIM_DURATION:
            self.finished = True

        return record

    def run_to_completion(self, dt: float = None) -> List[StepRecord]:
        """Run simulation until finished."""
        if dt is None:
            dt = cfg.SIM_DT
        while not self.finished:
            self.step(dt)
        return self.records

    def get_metrics(self) -> SimulationMetrics:
        """Compute metrics for the completed simulation."""
        nominal_dist = center_distance(
            self.scenario.ownship_start.x, self.scenario.ownship_start.y,
            self.scenario.ownship_goal_x, self.scenario.ownship_goal_y,
        )
        return compute_metrics(
            self.records,
            self.controller.name,
            self.scenario.name,
            nominal_dist,
        )


class MultiSimulator:
    """
    Runs the same scenario with multiple algorithms for comparison.
    """

    def __init__(self, scenario: ScenarioConfig,
                 controllers: List[BaseController],
                 params: USVParameters = None):
        self.scenario = scenario
        self.controllers = controllers
        self.params = params or USVParameters()
        self.simulators: List[Simulator] = []

        for ctrl in controllers:
            sim = Simulator(scenario, ctrl, self.params)
            self.simulators.append(sim)

    def reset_all(self):
        """Reset all simulators."""
        for sim in self.simulators:
            sim.reset()

    def step_all(self, dt: float = None) -> List[StepRecord]:
        """Step all simulators by one time step."""
        records = []
        for sim in self.simulators:
            rec = sim.step(dt)
            records.append(rec)
        return records

    @property
    def all_finished(self) -> bool:
        return all(sim.finished for sim in self.simulators)

    def run_all_to_completion(self, dt: float = None):
        """Run all simulators to completion."""
        if dt is None:
            dt = cfg.SIM_DT
        while not self.all_finished:
            self.step_all(dt)

    def get_all_metrics(self) -> List[SimulationMetrics]:
        """Get metrics for all simulators."""
        return [sim.get_metrics() for sim in self.simulators]
