"""
Performance metrics of a simulation run (paper Sec. VI-B).
"""

import math
from typing import List, Optional, Tuple

import data.config as cfg
from core.entities import StepRecord, SimulationMetrics

# Common cruise speed of all controllers
CRUISE_SPEED = cfg.CRUISE_SPEED

# Per-step rule-adherence predicate (paper Sec. VI-B)
ALTERATION_DEG = 3.0          # heading change counted as an alteration
SLOW_SPEED_FRACTION = 0.60    # commanded speed counted as a speed reduction
GIVE_WAY_TYPES = ("head_on", "crossing_give_way", "overtaking")

# Manoeuvre-commitment detection: consecutive non-hold_course steps
COMMIT_STEPS = 5


def _adherent(rec: StepRecord, goal: Tuple[float, float]) -> bool:
    """Common per-step rule-adherence predicate, applied to every controller
    from its commanded heading and speed: a commanded port alteration
    (> 3 deg to port of the goal heading) is non-adherent; otherwise a
    starboard alteration (> 3 deg), a speed below 60 % of cruise (including a
    stop), or course holding while the nearest target is beyond the avoidance
    domain is adherent."""
    s, cmd = rec.state, rec.command
    goal_heading = math.atan2(goal[1] - s.y, goal[0] - s.x)
    delta = math.atan2(math.sin(cmd.desired_heading - goal_heading),
                       math.cos(cmd.desired_heading - goal_heading))
    if delta > math.radians(ALTERATION_DEG):
        return False                                  # port alteration
    if delta < -math.radians(ALTERATION_DEG):
        return True                                   # starboard alteration
    if cmd.desired_speed < SLOW_SPEED_FRACTION * CRUISE_SPEED:
        return True                                   # speed reduction / stop
    return rec.min_distance > cfg.AVOIDANCE_DOMAIN    # course holding


def _adherent_from_label(rec: StepRecord) -> bool:
    """Fallback predicate from the controller-reported manoeuvre label, used
    only when no goal is given to :func:`compute_metrics`."""
    enc, man = rec.encounter.encounter_type, rec.command.manoeuvre
    if enc == "head_on" and man in ("early_stbd", "late_stbd", "slow_down"):
        return True
    if enc == "crossing_give_way" and man in ("early_stbd", "late_stbd", "slow_down", "stop"):
        return True
    if enc == "overtaking" and man in ("early_stbd", "late_stbd", "slow_down"):
        return True
    return man == "hold_course" and rec.min_distance > cfg.AVOIDANCE_DOMAIN


def compute_metrics(records: List[StepRecord], algorithm: str,
                    scenario: str, nominal_distance: float,
                    goal: Optional[Tuple[float, float]] = None) -> SimulationMetrics:
    """
    Compute all performance metrics from the step records of one run.

    Args:
        records: step records from :class:`core.simulator.Simulator`
        algorithm: controller name
        scenario: scenario name
        nominal_distance: straight-line start-to-goal distance
        goal: goal position, used by the rule-adherence predicate

    Returns:
        SimulationMetrics with all fields populated
    """
    m = SimulationMetrics(algorithm=algorithm, scenario=scenario)
    if not records:
        return m

    m.sim_duration = records[-1].time - records[0].time if len(records) > 1 else 0.0
    m.nominal_distance = nominal_distance

    path_length = 0.0
    speeds = []
    yaw_rates = []
    thrust_integral = 0.0
    yaw_activity = 0.0
    peak_diff = 0.0
    min_sep = float('inf')
    has_collision = False

    # COLREG adherence is scored on give-way steps only
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

        if rec.min_distance < min_sep:
            min_sep = rec.min_distance

        if rec.collision:
            has_collision = True

        if rec.thruster is not None:
            thrust_integral += (abs(rec.thruster.T_L) + abs(rec.thruster.T_R)) * dt
            diff = abs(rec.thruster.T_R - rec.thruster.T_L)
            if diff > peak_diff:
                peak_diff = diff
            yaw_activity += abs(rec.state.r if rec.state else 0.0) * dt

        if rec.encounter is not None and rec.command is not None:
            if rec.encounter.encounter_type in GIVE_WAY_TYPES:
                give_way_steps += 1
                if goal is not None and rec.state is not None:
                    compliant_steps += _adherent(rec, goal)
                else:
                    compliant_steps += _adherent_from_label(rec)

    m.collision_flag = has_collision
    m.min_separation = min_sep
    m.path_length = path_length
    m.path_efficiency = nominal_distance / max(path_length, 0.01)
    m.avg_speed = sum(speeds) / max(len(speeds), 1)
    m.peak_yaw_rate = max(yaw_rates) if yaw_rates else 0.0
    m.total_thrust_integral = thrust_integral
    m.total_yaw_activity = yaw_activity
    m.peak_diff_thrust = peak_diff
    m.colreg_compliance = compliant_steps / give_way_steps if give_way_steps > 0 else 1.0
    m.manoeuvre_commit_time = _compute_commit_time(records)
    return m


def _compute_commit_time(records: List[StepRecord]) -> float:
    """
    Time before CPA at which a sustained avoidance manoeuvre begins.

    CPA is the step of minimum clearance; the manoeuvre starts at the first
    run of COMMIT_STEPS consecutive non-hold_course commands.  Returns inf
    when no such run exists.
    """
    min_d = float('inf')
    cpa_time = 0.0
    for rec in records:
        if rec.min_distance < min_d:
            min_d = rec.min_distance
            cpa_time = rec.time

    consecutive = 0
    commit_time = float('inf')
    for rec in records:
        if rec.command and rec.command.manoeuvre != "hold_course":
            consecutive += 1
            if consecutive >= COMMIT_STEPS and commit_time == float('inf'):
                commit_time = cpa_time - rec.time
        else:
            consecutive = 0

    return max(commit_time, 0.0)
