"""
COLREG encounter classification utilities.

Classifies encounters between ownship and targets based on
relative bearing, relative heading, and closing geometry.
"""

import math
from core.entities import VesselState, ObstacleState, EncounterInfo
from core.geometry import (
    bearing_from_to, wrap_angle, center_distance, compute_dcpa_tcpa
)
import data.config as cfg


def classify_encounter(own: VesselState, tgt: ObstacleState) -> EncounterInfo:
    """
    Classify the encounter type between ownship and a target.

    Returns an EncounterInfo with type, bearings, distances, DCPA/TCPA.
    """
    dist = center_distance(own.x, own.y, tgt.x, tgt.y)

    # Relative bearing from ownship to target
    rel_bearing = bearing_from_to(own.x, own.y, own.psi, tgt.x, tgt.y)

    # Relative heading difference
    rel_heading = wrap_angle(tgt.psi - own.psi)

    # DCPA / TCPA
    dcpa, tcpa = compute_dcpa_tcpa(
        own.x, own.y, own.vx, own.vy,
        tgt.x, tgt.y, tgt.vx, tgt.vy
    )

    # Relative speed (closing speed)
    dvx = own.vx - tgt.vx
    dvy = own.vy - tgt.vy
    closing_speed = math.sqrt(dvx**2 + dvy**2)

    # Determine side
    if abs(rel_bearing) < math.radians(10):
        side = "ahead"
    elif rel_bearing > 0:
        side = "port"  # target is to port
    else:
        side = "starboard"  # target is to starboard

    # Correct: in maritime convention, if target is on our starboard and
    # approaching, we may need to give way. Let's use standard convention:
    # positive bearing = starboard in maritime, but here we use math convention
    # Let's re-derive: bearing > 0 means target is above (port in screen-north-up)
    # For the algorithm logic we use the raw bearing; encounter type is what matters.

    # Classification
    encounter_type = "none"

    if dist > cfg.AVOIDANCE_DOMAIN * 3:
        encounter_type = "none"
    else:
        # Target is ahead (within ±90 deg)
        target_ahead = abs(rel_bearing) < math.radians(90)

        # Near reciprocal headings (head-on): heading difference ~180 deg
        is_reciprocal = abs(abs(rel_heading) - math.pi) < math.radians(20)

        # Overtaking: ownship approaches from behind the target
        bearing_from_target = bearing_from_to(tgt.x, tgt.y, tgt.psi, own.x, own.y)
        ownship_behind_target = abs(bearing_from_target) > math.radians(112.5)

        if is_reciprocal and target_ahead and abs(rel_bearing) < math.radians(15):
            encounter_type = "head_on"
        elif ownship_behind_target and own.u > tgt.speed:
            encounter_type = "overtaking"
        elif target_ahead:
            # Crossing situation
            # If target is on our starboard side (rel_bearing < 0 in our convention),
            # we are give-way vessel
            if rel_bearing < 0:
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
