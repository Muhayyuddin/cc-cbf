"""
Scenario generators for the collision avoidance simulator.

All scenarios are scaled for a 6 m class USV with max speed ~4.1 m/s.
"""

import math
import copy
import random
from typing import List
from core.entities import VesselState, ObstacleState, ScenarioConfig
import data.config as cfg


def create_head_on_scenario(seed: int = 0, offset: float = 0.0,
                            speed_var: float = 0.0,
                            heading_var: float = 0.0) -> ScenarioConfig:
    """
    Head-on encounter: ownship and target approach on reciprocal headings.

    Ownship starts at left, heading east. Target starts at right, heading west.
    """
    rng = random.Random(seed)

    own_speed = 3.0 + rng.gauss(0, speed_var)
    tgt_speed = 2.5 + rng.gauss(0, speed_var)
    lat_offset = offset + rng.gauss(0, 2.0) if offset != 0 else rng.gauss(0, 1.0)

    ownship = VesselState(
        x=-100.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0
    )

    target = ObstacleState(
        x=100.0, y=0.0,
        psi=math.pi + rng.gauss(0, heading_var),
        speed=tgt_speed,
        length=8.0, width=3.0,
        label="T1", vessel_type="power_driven"
    )

    return ScenarioConfig(
        name="head_on",
        ownship_start=ownship,
        ownship_goal_x=100.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=[target],
        description="Head-on encounter with reciprocal headings"
    )


def create_crossing_give_way_scenario(seed: int = 0, offset: float = 0.0,
                                       speed_var: float = 0.0,
                                       heading_var: float = 0.0) -> ScenarioConfig:
    """
    Crossing give-way: target approaches from ownship's starboard side.

    Ownship heads east, target heads north from below-right.
    """
    rng = random.Random(seed)

    own_speed = 3.0 + rng.gauss(0, speed_var)
    tgt_speed = 2.5 + rng.gauss(0, speed_var)
    lat_offset = rng.gauss(0, 1.0)

    ownship = VesselState(
        x=-100.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0
    )

    # Target from starboard: below and to the right, heading north (pi/2)
    target = ObstacleState(
        x=0.0 + rng.gauss(0, 3.0),
        y=-100.0,
        psi=math.pi / 2.0 + rng.gauss(0, heading_var),
        speed=tgt_speed,
        length=8.0, width=3.0,
        label="T1", vessel_type="power_driven"
    )

    return ScenarioConfig(
        name="crossing_give_way",
        ownship_start=ownship,
        ownship_goal_x=100.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=[target],
        description="Crossing give-way: target from starboard"
    )


def create_overtaking_scenario(seed: int = 0, offset: float = 0.0,
                                speed_var: float = 0.0,
                                heading_var: float = 0.0) -> ScenarioConfig:
    """
    Overtaking: ownship faster, approaching a slower vessel from behind.
    """
    rng = random.Random(seed)

    own_speed = 3.5 + rng.gauss(0, speed_var)
    tgt_speed = 1.5 + rng.gauss(0, speed_var * 0.5)
    lat_offset = rng.gauss(0, 1.0)

    ownship = VesselState(
        x=-100.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0
    )

    # Target ahead, same direction, slower — placed closer for realistic
    # overtaking encounter within the 200 m corridor
    target = ObstacleState(
        x=-50.0,
        y=0.0 + rng.gauss(0, 2.0),
        psi=0.0 + rng.gauss(0, heading_var),
        speed=tgt_speed,
        length=10.0, width=4.0,
        label="T1", vessel_type="power_driven"
    )

    return ScenarioConfig(
        name="overtaking",
        ownship_start=ownship,
        ownship_goal_x=100.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=[target],
        description="Overtaking a slower vessel"
    )


def create_static_obstacle_field(seed: int = 0, offset: float = 0.0,
                                  speed_var: float = 0.0,
                                  heading_var: float = 0.0) -> ScenarioConfig:
    """
    Static obstacle field: channel with fixed rectangular obstacles.
    """
    rng = random.Random(seed)

    own_speed = 2.5 + rng.gauss(0, speed_var)
    lat_offset = rng.gauss(0, 1.0)

    ownship = VesselState(
        x=-100.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0
    )

    obstacles = []
    # Create a corridor of obstacles spread over 200 m
    positions = [
        (-50.0, 15.0, math.radians(30)),
        (-15.0, -12.0, math.radians(-20)),
        (20.0, 10.0, math.radians(45)),
        (55.0, -15.0, math.radians(-10)),
        (85.0, 8.0, math.radians(15)),
    ]
    for i, (ox, oy, opsi) in enumerate(positions):
        ox += rng.gauss(0, 2.0)
        oy += rng.gauss(0, 2.0)
        obstacles.append(ObstacleState(
            x=ox, y=oy, psi=opsi, speed=0.0,
            length=8.0 + rng.uniform(-2, 2),
            width=4.0 + rng.uniform(-1, 1),
            is_static=True,
            label=f"O{i+1}",
            vessel_type="restricted"
        ))

    return ScenarioConfig(
        name="static_obstacles",
        ownship_start=ownship,
        ownship_goal_x=100.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=obstacles,
        context_flags={'narrow_channel': True, 'reduced_visibility': False},
        description="Static obstacle field in corridor"
    )


def create_head_on_then_overtaking_scenario(seed: int = 0, offset: float = 0.0,
                                             speed_var: float = 0.0,
                                             heading_var: float = 0.0) -> ScenarioConfig:
    """
    Sequential multi-vessel encounter: head-on then overtaking.

    Layout (all moving east = positive x direction):

        T1 (head-on)  ←←←  approaching fast from x=+100, heading west
        USV           →→→  starts at x=-150, heading east
        T2 (overtaking)→→  ahead of USV at x=+50, heading east but slow

    Sequence of events:
      1. USV closes with T1 (head-on, Rule 14) — avoids to starboard, passes
      2. T1 moves away, T2 enters activation range — USV catches up (overtaking,
         Rule 13) — avoids to starboard, passes
      3. USV resumes course to goal at x=+350

    Separation between T1 and T2 (~150 m) ensures only one is active at a time
    within ACTIVATION_RANGE = 95 m.
    """
    rng = random.Random(seed)

    own_speed  = 3.0 + rng.gauss(0, speed_var)
    t1_speed   = 3.5 + rng.gauss(0, speed_var)   # T1 faster, head-on
    t2_speed   = 0.5 + rng.gauss(0, speed_var * 0.5)  # T2 very slow, same direction

    lat_offset = offset + rng.gauss(0, 1.0)

    ownship = VesselState(
        x=-150.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0,
    )

    # T1: head-on, coming from the east at speed 3.5 m/s
    # Placed at x=+100 so USV meets it after ~70 m of travel (~23 s)
    t1 = ObstacleState(
        x=+100.0, y=0.0 + rng.gauss(0, 1.0),
        psi=math.pi + rng.gauss(0, heading_var),   # heading west
        speed=t1_speed,
        length=8.0, width=3.0,
        label="T1_HeadOn", vessel_type="power_driven",
    )

    # T2: overtaking target — ahead of USV, heading east, slow
    # Placed at x=+50; USV reaches it after T1 is cleared (~x=+150 onward)
    # By then USV has passed T1 and T2 is still ahead, entering activation range
    t2 = ObstacleState(
        x=+50.0, y=0.0 + rng.gauss(0, 1.5),
        psi=0.0 + rng.gauss(0, heading_var),        # heading east (same as USV)
        speed=t2_speed,
        length=10.0, width=4.0,
        label="T2_Overtaking", vessel_type="power_driven",
    )

    return ScenarioConfig(
        name="head_on_then_overtaking",
        ownship_start=ownship,
        ownship_goal_x=350.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=[t1, t2],
        description=(
            "Sequential encounter: head-on (Rule 14) with T1 approaching fast, "
            "then overtaking (Rule 13) of slower T2 ahead. "
            "USV returns to nominal path between encounters."
        ),
    )


def create_parallel_head_on_scenario(seed: int = 0, offset: float = 0.0,
                                      speed_var: float = 0.0,
                                      heading_var: float = 0.0) -> ScenarioConfig:
    """
    Simultaneous multi-vessel encounter: two vessels approaching in parallel.

    Layout:
        T1 ←←← (y=+14 m, heading west)
        USV →→→ (y=0)
        T2 ←←← (y=-14 m, heading west)

    Both vessels enter the USV's activation range at the same time, creating
    two simultaneous CBF constraints. The QP finds the minimum-intervention
    safe velocity — typically slowing and threading the gap between them.
    Demonstrates the analytical QP handling simultaneous conflicting constraints.

    Lateral offset of ±14 m keeps both within the 15° head-on bearing
    tolerance at 100 m range (atan2(14,100) ≈ 8°), ensuring Rule 14
    classification for both.
    """
    rng = random.Random(seed)

    own_speed = 3.0 + rng.gauss(0, speed_var)
    t_speed   = 1.5 + rng.gauss(0, speed_var)

    lat_offset = offset  # keep symmetric — no random lat offset

    ownship = VesselState(
        x=-100.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0,
    )

    # T1: above — approaching head-on from upper lane
    t1 = ObstacleState(
        x=+100.0, y=+14.0 + lat_offset + rng.gauss(0, 0.3),
        psi=math.pi + rng.gauss(0, heading_var),   # heading west
        speed=t_speed + rng.gauss(0, speed_var * 0.3),
        length=8.0, width=3.0,
        label="T1_Upper", vessel_type="power_driven",
    )

    # T2: below — approaching head-on from lower lane
    t2 = ObstacleState(
        x=+100.0, y=-14.0 + lat_offset + rng.gauss(0, 0.3),
        psi=math.pi + rng.gauss(0, heading_var),   # heading west
        speed=t_speed + rng.gauss(0, speed_var * 0.3),
        length=8.0, width=3.0,
        label="T2_Lower", vessel_type="power_driven",
    )

    return ScenarioConfig(
        name="parallel_head_on",
        ownship_start=ownship,
        ownship_goal_x=100.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=[t1, t2],
        description=(
            "Two vessels approaching in parallel lanes (±14 m offset). "
            "Simultaneous CBF constraints — QP finds safe gap between them. "
            "Tests multi-obstacle constraint stacking (Rule 14 × 2)."
        ),
    )


def create_mixed_rules_scenario(seed: int = 0, offset: float = 0.0,
                                 speed_var: float = 0.0,
                                 heading_var: float = 0.0) -> ScenarioConfig:
    """
    Simultaneous mixed-rule encounter: head-on (Rule 14) + overtaking (Rule 13).

    Layout:
        T1 ←←← (x=+80, heading west — head-on, Rule 14)
        USV →→→ (x=-100, heading east)
        T2 →→   (x=-30,  heading east, very slow — overtaking, Rule 13)

    Sequence:
      1. T2 is immediately ahead and slow → overtaking CBF activates first
      2. As USV manoeuvres around T2, T1 closes → head-on CBF joins
      3. Both constraints active simultaneously while USV is between them
      4. USV clears both and resumes course to goal

    T1 placed at x=+80 (180 m from USV start) so it enters LiDAR range
    (~100 m) around t=25 s, while T2 avoidance is still ongoing.
    Demonstrates Rules 13 + 14 handled by the same CC-CBF simultaneously.
    """
    rng = random.Random(seed)

    own_speed = 3.2 + rng.gauss(0, speed_var)
    t1_speed  = 2.8 + rng.gauss(0, speed_var)    # head-on vessel, moderate speed
    t2_speed  = 0.8 + rng.gauss(0, speed_var * 0.5)  # very slow overtaking target

    lat_offset = offset + rng.gauss(0, 1.0)

    ownship = VesselState(
        x=-100.0, y=0.0 + lat_offset,
        psi=0.0 + rng.gauss(0, heading_var),
        u=own_speed, r=0.0,
    )

    # T1: head-on — placed slightly south of the centreline (y=-8) to match
    # the USV's expected y-position (~-8 to -12 m) when T1 enters range.
    # The USV dodges south past T2, so T1 must be on that same deflected path
    # to be classified as head-on (bearing < 15°) rather than crossing.
    t1 = ObstacleState(
        x=+80.0, y=-8.0,
        psi=math.pi + rng.gauss(0, heading_var),   # heading west
        speed=t1_speed,
        length=8.0, width=3.0,
        label="T1_HeadOn", vessel_type="power_driven",
    )

    # T2: very slow vessel immediately ahead, on the centreline — USV overtakes
    # (Rule 13) and naturally passes to starboard (south), moving toward T1's line.
    t2 = ObstacleState(
        x=-30.0, y=0.0 + rng.gauss(0, 0.5),
        psi=0.0 + rng.gauss(0, heading_var),        # heading east (same as USV)
        speed=t2_speed,
        length=10.0, width=4.0,
        label="T2_Slow", vessel_type="power_driven",
    )

    return ScenarioConfig(
        name="mixed_rules",
        ownship_start=ownship,
        ownship_goal_x=100.0,
        ownship_goal_y=0.0 + lat_offset,
        targets=[t1, t2],
        description=(
            "Mixed simultaneous rules: Rule 13 (overtaking T2 slow ahead) "
            "and Rule 14 (head-on T1 approaching) handled by the same CC-CBF. "
            "Both CBF constraints briefly active at once."
        ),
    )


SCENARIO_GENERATORS = {
    'head_on':                   create_head_on_scenario,
    'crossing_give_way':         create_crossing_give_way_scenario,
    'overtaking':                create_overtaking_scenario,
    'static_obstacles':          create_static_obstacle_field,
    'head_on_then_overtaking':   create_head_on_then_overtaking_scenario,
    'parallel_head_on':          create_parallel_head_on_scenario,
    'mixed_rules':               create_mixed_rules_scenario,
}


def get_scenario(name: str, seed: int = 0, offset: float = 0.0,
                 speed_var: float = 0.0, heading_var: float = 0.0) -> ScenarioConfig:
    """Get a scenario by name with optional randomization."""
    gen = SCENARIO_GENERATORS.get(name)
    if gen is None:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_GENERATORS.keys())}")
    return gen(seed=seed, offset=offset, speed_var=speed_var, heading_var=heading_var)
