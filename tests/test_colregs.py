import math
import random

import pytest

from core.colregs import classify_encounter, classify_type_fast
from core.entities import ObstacleState, VesselState

TYPES = [None, "none", "head_on", "crossing_give_way", "crossing_stand_on", "overtaking"]


def own(x=0.0, y=0.0, psi=0.0, u=3.0):
    return VesselState(x=x, y=y, psi=psi, u=u)


def tgt(x, y, psi, speed=2.5, static=False):
    return ObstacleState(x=x, y=y, psi=psi, speed=speed, is_static=static)


@pytest.mark.parametrize("target, expected", [
    (tgt(60, 0, math.pi), "head_on"),                         # reciprocal, dead ahead
    (tgt(40, -40, math.pi / 2), "crossing_give_way"),         # from starboard
    (tgt(40, 40, -math.pi / 2), "crossing_stand_on"),         # from port
    (tgt(30, 0, 0.0, speed=1.0), "overtaking"),               # slower, same course ahead
    (tgt(-30, 0, 0.0, speed=1.0), "none"),                    # astern, opening
    (tgt(200, 0, math.pi), "none"),                           # beyond d_act
    (tgt(30, 0, 0.0, speed=0.0, static=True), "none"),        # static obstacle
])
def test_canonical_classes(target, expected):
    assert classify_encounter(own(), target).encounter_type == expected
    assert classify_type_fast(own(), target) == expected


def test_fast_classifier_matches_reference():
    rng = random.Random(0)
    for _ in range(20000):
        o = own(rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-math.pi, math.pi),
                rng.uniform(0, 4))
        t = tgt(rng.uniform(-120, 120), rng.uniform(-120, 120), rng.uniform(-math.pi, math.pi),
                rng.uniform(0, 4), static=rng.random() < 0.05)
        prev = rng.choice(TYPES)
        assert (classify_type_fast(o, t, prev)
                == classify_encounter(o, t, prev_type=prev, light=True).encounter_type
                == classify_encounter(o, t, prev_type=prev).encounter_type)


def test_hysteresis_keeps_incumbent_class_inside_dead_band():
    # Target bearing 16 deg: outside the 15 deg head-on sector for a fresh
    # classification, inside the widened sector while head-on is incumbent.
    b = math.radians(16)
    t = tgt(80 * math.cos(b), 80 * math.sin(b), math.pi)
    assert classify_encounter(own(), t).encounter_type != "head_on"
    assert classify_encounter(own(), t, prev_type="head_on").encounter_type == "head_on"
