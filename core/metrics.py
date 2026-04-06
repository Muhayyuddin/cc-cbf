"""
Performance metrics computation for simulation runs.
"""

import math
from typing import List
from core.entities import (
    StepRecord, SimulationMetrics, VesselState, ObstacleState, EncounterInfo
)
from core.geometry import wrap_angle
import data.config as cfg


def compute_metrics(records: List[StepRecord], algorithm: str,
                    scenario: str, nominal_distance: float) -> SimulationMetrics:
    """
    Compute all performance metrics from a list of step records.

    Args:
        records: list of StepRecord from simulation
        algorithm: algorithm name string
        scenario: scenario name string
        nominal_distance: straight-line distance ownship should travel

    Returns:
        SimulationMetrics with all fields populated
    """
    if not records:
        return SimulationMetrics(algorithm=algorithm, scenario=scenario)

    m = SimulationMetrics(algorithm=algorithm, scenario=scenario)

    # Time
    m.sim_duration = records[-1].time - records[0].time if len(records) > 1 else 0.0
    m.nominal_distance = nominal_distance

    # Path length
    path_length = 0.0
    speeds = []
    yaw_rates = []
    thrust_integral = 0.0
    yaw_activity = 0.0
    peak_diff = 0.0
    min_sep = float('inf')
    has_collision = False
    llm_queries = 0

    # COLREG compliance tracking
    compliant_steps = 0
    give_way_steps = 0

    dt = cfg.SIM_DT

    for i, rec in enumerate(records):
        if rec.state is not None:
            speeds.append(rec.state.u)
            yaw_rates.append(abs(rec.state.r))

        if i > 0 and records[i].state is not None and records[i - 1].state is not None:
            s0 = records[i - 1].state
            s1 = records[i].state
            dx = s1.x - s0.x
            dy = s1.y - s0.y
            path_length += math.sqrt(dx * dx + dy * dy)

        # Min distance
        if rec.min_distance < min_sep:
            min_sep = rec.min_distance

        # Collision
        if rec.collision:
            has_collision = True

        # LLM queries
        if rec.llm_queried:
            llm_queries += 1

        # Thrust integral
        if rec.thruster is not None:
            thrust_integral += (abs(rec.thruster.T_L) + abs(rec.thruster.T_R)) * dt
            diff = abs(rec.thruster.T_R - rec.thruster.T_L)
            if diff > peak_diff:
                peak_diff = diff
            yaw_activity += abs(rec.state.r if rec.state else 0.0) * dt

        # COLREG compliance
        if rec.encounter is not None and rec.command is not None:
            enc = rec.encounter
            cmd = rec.command
            if enc.encounter_type in ("head_on", "crossing_give_way", "overtaking"):
                give_way_steps += 1
                # Check if manoeuvre is appropriate
                if enc.encounter_type == "head_on" and cmd.manoeuvre in ("early_stbd", "late_stbd", "slow_down"):
                    compliant_steps += 1
                elif enc.encounter_type == "crossing_give_way" and cmd.manoeuvre in ("early_stbd", "late_stbd", "slow_down", "stop"):
                    compliant_steps += 1
                elif enc.encounter_type == "overtaking" and cmd.manoeuvre in ("early_stbd", "late_stbd", "slow_down"):
                    compliant_steps += 1
                elif cmd.manoeuvre == "hold_course":
                    # Holding course when give-way = non-compliant, unless far away
                    if rec.min_distance > cfg.AVOIDANCE_DOMAIN:
                        compliant_steps += 1  # far enough, no action needed yet

    m.collision_flag = has_collision
    m.min_separation = min_sep
    m.path_length = path_length
    m.path_efficiency = nominal_distance / max(path_length, 0.01)
    m.total_llm_queries = llm_queries
    m.avg_speed = sum(speeds) / max(len(speeds), 1)
    m.peak_yaw_rate = max(yaw_rates) if yaw_rates else 0.0
    m.total_thrust_integral = thrust_integral
    m.total_yaw_activity = yaw_activity
    m.peak_diff_thrust = peak_diff

    # COLREG compliance score
    if give_way_steps > 0:
        m.colreg_compliance = compliant_steps / give_way_steps
    else:
        m.colreg_compliance = 1.0

    # Manoeuvre commitment time - find first sustained non-hold_course manoeuvre
    m.manoeuvre_commit_time = _compute_commit_time(records)

    return m


def _compute_commit_time(records: List[StepRecord]) -> float:
    """
    Estimate time before CPA when a consistent avoidance manoeuvre begins.

    Looks for the first sustained (>=5 consecutive steps) non-hold_course command.
    """
    # Find approximate CPA time (time of minimum distance)
    min_d = float('inf')
    cpa_time = 0.0
    for rec in records:
        if rec.min_distance < min_d:
            min_d = rec.min_distance
            cpa_time = rec.time

    # Find first sustained manoeuvre
    consecutive = 0
    commit_time = float('inf')
    threshold = 5  # steps

    for rec in records:
        if rec.command and rec.command.manoeuvre != "hold_course":
            consecutive += 1
            if consecutive >= threshold and commit_time == float('inf'):
                commit_time = cpa_time - rec.time
        else:
            consecutive = 0

    return max(commit_time, 0.0)
