"""
COLREG encounter classification utilities.

Classifies encounters between ownship and targets based on
relative bearing, relative heading, and closing geometry.
"""

import math
from typing import Optional

from core.entities import VesselState, ObstacleState, EncounterInfo
from core.geometry import (
    bearing_from_to, wrap_angle, center_distance, compute_dcpa_tcpa
)

# Classification dead-band Delta_hys (paper, Remark "Classification
# hysteresis"): a new encounter type is accepted only after the relevant
# angle has entered the new sector by HYSTERESIS_DEG.  Every angular
# threshold below is widened by this amount in favour of the incumbent
# class *prev_type* and narrowed against it.  The memoryless classifier
# (prev_type=None, or HYSTERESIS_DEG=0) is what the simulator hands to
# every controller and what the compliance metrics are scored against;
# CC-CBF uses the hysteretic variant to build its own switching signal tau.
HYSTERESIS_DEG = 2.0

# Sector thresholds of paper Sec. III-D
HEAD_ON_DPSI_DEG = 15.0      # |wrap(dpsi - pi)| tolerance for head-on
HEAD_ON_BEARING_DEG = 15.0   # |theta_rel| tolerance for head-on
CROSSING_SECTOR_DEG = 112.5  # crossing sectors: |theta_rel| < 112.5 deg
OVERTAKING_SECTOR_DEG = 112.5  # own ship more than 22.5 deg abaft the target's beam
# Static obstacles are not vessels: no COLREG encounter class
STATIC_AS_NONE = True
# Encounters are classified inside the activation distance d_act (Sec. V-B)
CLASSIFICATION_RANGE = 95.0
# Dead-band on the range rate of the overtaking closing test (m/s)
CLOSING_DEADBAND_MPS = 0.5


def classify_encounter(own: VesselState, tgt: ObstacleState,
                       prev_type: Optional[str] = None,
                       light: bool = False) -> EncounterInfo:
    """
    Classify the encounter between the own ship and a target.

    *prev_type* is the class assigned to this target at the previous step.
    When given (and HYSTERESIS_DEG > 0) the angular thresholds are biased
    in favour of that class, implementing the dead-band Delta_hys.
    *light* skips the DCPA/TCPA computation (the class is unchanged).

    Returns an EncounterInfo with type, bearing, distance and DCPA/TCPA.
    """
    dist = center_distance(own.x, own.y, tgt.x, tgt.y)
    if light and ((STATIC_AS_NONE and getattr(tgt, "is_static", False))
                  or dist > CLASSIFICATION_RANGE):
        # same class as the full path below; CPA fields are not needed
        return EncounterInfo(encounter_type="none", distance=dist,
                             target_label=tgt.label)

    # Relative bearing from ownship to target
    rel_bearing = bearing_from_to(own.x, own.y, own.psi, tgt.x, tgt.y)

    # Relative heading difference
    rel_heading = wrap_angle(tgt.psi - own.psi)

    # DCPA / TCPA (not needed by the controller's own classification)
    if light:
        dcpa, tcpa = float("inf"), float("inf")
    else:
        dcpa, tcpa = compute_dcpa_tcpa(
            own.x, own.y, own.vx, own.vy,
            tgt.x, tgt.y, tgt.vx, tgt.vy
        )

    # Relative speed (closing speed)
    dvx = own.vx - tgt.vx
    dvy = own.vy - tgt.vy
    closing_speed = math.sqrt(dvx**2 + dvy**2)
    # Range rate  r_dot = (p_o - p_t) . (v_o - v_t) / d  (negative = closing).
    # Overtaking requires a closing speed of at least CLOSING_DEADBAND_MPS
    # (both classifiers); the hysteretic classifier keeps an incumbent
    # overtaking class while r_dot < +CLOSING_DEADBAND_MPS, so a target
    # trailed at almost constant range does not toggle the class.
    r_dot = ((own.x - tgt.x) * dvx + (own.y - tgt.y) * dvy) / max(dist, 1e-6)
    if prev_type == "overtaking":
        closing = r_dot < CLOSING_DEADBAND_MPS
    else:
        closing = r_dot < -CLOSING_DEADBAND_MPS

    # Side of the target (bearings are counter-clockwise positive)
    if abs(rel_bearing) < math.radians(10):
        side = "ahead"
    elif rel_bearing > 0:
        side = "port"  # target is to port
    else:
        side = "starboard"  # target is to starboard

    # Classification (angular thresholds biased by the dead-band Delta_hys
    # in favour of the incumbent class *prev_type*; range and speed tests
    # are memoryless)
    encounter_type = "none"

    hys = math.radians(HYSTERESIS_DEG) if prev_type is not None else 0.0
    inc = prev_type or "none"

    def _band(base, favours_incumbent):
        """Widen a threshold when staying in the incumbent class, narrow it
        when leaving; the resulting dead-band is 2 * Delta_hys wide."""
        return base + hys if favours_incumbent else base - hys

    crossing_like = ("head_on", "crossing_give_way", "crossing_stand_on")

    if STATIC_AS_NONE and getattr(tgt, "is_static", False):
        # COLREG encounter rules apply between vessels; a static obstacle
        # is handled by the barrier without an encounter class.
        encounter_type = "none"
    elif dist > CLASSIFICATION_RANGE:
        encounter_type = "none"
    else:
        # Target forward of 22.5 deg abaft the beam (|theta_rel| < 112.5 deg),
        # the crossing sectors of paper Sec. III-D
        target_ahead = abs(rel_bearing) < _band(math.radians(CROSSING_SECTOR_DEG),
                                                inc in crossing_like)

        # Near reciprocal headings (head-on): |wrap(dpsi - pi)| < 15 deg
        is_reciprocal = abs(abs(rel_heading) - math.pi) < _band(
            math.radians(HEAD_ON_DPSI_DEG), inc == "head_on")

        # Overtaking: ownship approaches from behind the target
        bearing_from_target = bearing_from_to(tgt.x, tgt.y, tgt.psi, own.x, own.y)
        ownship_behind_target = abs(bearing_from_target) > _band(
            math.radians(OVERTAKING_SECTOR_DEG), inc != "overtaking")

        head_on_thr = _band(math.radians(HEAD_ON_BEARING_DEG), inc == "head_on")

        if is_reciprocal and target_ahead and abs(rel_bearing) < head_on_thr:
            encounter_type = "head_on"
        elif ownship_behind_target and closing:
            # Rule 13 "coming up with": positive closing speed (range
            # decreasing), paper Sec. III-D
            encounter_type = "overtaking"
        elif target_ahead:
            # Crossing situation: the give-way / stand-on split sits at
            # rel_bearing = 0 (target to starboard => ownship gives way); a
            # +-Delta_hys dead-band around it retains the incumbent role.
            if inc == "crossing_give_way":
                split = hys
            elif inc == "crossing_stand_on":
                split = -hys
            else:
                split = 0.0
            if rel_bearing < split:
                encounter_type = "crossing_give_way"
            else:
                encounter_type = "crossing_stand_on"
        else:
            encounter_type = "none"

    return EncounterInfo(
        encounter_type=encounter_type,
        relative_bearing=rel_bearing,
        relative_speed=closing_speed,
        distance=dist,
        dcpa=dcpa,
        tcpa=tcpa,
        target_label=tgt.label,
        target_side=side,
    )


def is_give_way(encounter_type: str) -> bool:
    """Check if ownship is the give-way vessel in this encounter."""
    return encounter_type in ("head_on", "crossing_give_way", "overtaking")


def classify_type_fast(own, tgt, prev_type=None) -> str:
    """Encounter class only (no CPA fields).

    Same tests and thresholds as :func:`classify_encounter`, written with
    scalar arithmetic for the CC-CBF control step (the equivalence is
    checked in ``tests/test_colregs.py``).
    """
    dx = tgt.x - own.x
    dy = tgt.y - own.y
    dist = math.sqrt(dx * dx + dy * dy)
    if (STATIC_AS_NONE and getattr(tgt, "is_static", False)) or dist > CLASSIFICATION_RANGE:
        return "none"
    rel_bearing = wrap_angle(math.atan2(dy, dx) - own.psi)
    rel_heading = wrap_angle(tgt.psi - own.psi)
    cpo, spo = math.cos(own.psi), math.sin(own.psi)
    cpt, spt = math.cos(tgt.psi), math.sin(tgt.psi)
    dvx = own.u * cpo - tgt.speed * cpt
    dvy = own.u * spo - tgt.speed * spt
    r_dot = (-dx * dvx + -dy * dvy) / max(dist, 1e-6)
    if prev_type == "overtaking":
        closing = r_dot < CLOSING_DEADBAND_MPS
    else:
        closing = r_dot < -CLOSING_DEADBAND_MPS
    hys = math.radians(HYSTERESIS_DEG) if prev_type is not None else 0.0
    inc = prev_type or "none"
    cross = math.radians(CROSSING_SECTOR_DEG)
    ab = abs(rel_bearing)
    target_ahead = ab < (cross + hys if inc in ("head_on", "crossing_give_way", "crossing_stand_on") else cross - hys)
    hon = math.radians(HEAD_ON_DPSI_DEG)
    is_reciprocal = abs(abs(rel_heading) - math.pi) < (hon + hys if inc == "head_on" else hon - hys)
    bft = wrap_angle(math.atan2(-dy, -dx) - tgt.psi)
    ot = math.radians(OVERTAKING_SECTOR_DEG)
    behind = abs(bft) > (ot + hys if inc != "overtaking" else ot - hys)
    h15 = math.radians(HEAD_ON_BEARING_DEG)
    if is_reciprocal and target_ahead and ab < (h15 + hys if inc == "head_on" else h15 - hys):
        return "head_on"
    if behind and closing:
        return "overtaking"
    if target_ahead:
        split = hys if inc == "crossing_give_way" else (-hys if inc == "crossing_stand_on" else 0.0)
        return "crossing_give_way" if rel_bearing < split else "crossing_stand_on"
    return "none"
