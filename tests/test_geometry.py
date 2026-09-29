import math

import pytest

from core.geometry import (
    bearing_from_to, compute_dcpa_tcpa, min_distance_rects, rect_overlap, wrap_angle,
)


@pytest.mark.parametrize("angle", [0.0, 1.0, -1.0, 3.5, -3.5, 7.0, -20.0, 100.0])
def test_wrap_angle_range_and_equivalence(angle):
    w = wrap_angle(angle)
    assert -math.pi <= w <= math.pi
    assert math.isclose(math.cos(w), math.cos(angle), abs_tol=1e-12)
    assert math.isclose(math.sin(w), math.sin(angle), abs_tol=1e-12)


def test_bearing_is_positive_to_port():
    # Heading east; a point to the north is on the port side
    assert bearing_from_to(0, 0, 0.0, 0, 10) > 0
    assert bearing_from_to(0, 0, 0.0, 0, -10) < 0
    assert math.isclose(bearing_from_to(0, 0, 0.0, 10, 0), 0.0)


def test_dcpa_tcpa_head_on():
    # Reciprocal courses 2 m apart laterally, closing at 5 m/s over 100 m
    dcpa, tcpa = compute_dcpa_tcpa(0, 0, 2.5, 0, 100, 2, -2.5, 0)
    assert math.isclose(dcpa, 2.0, abs_tol=1e-9)
    assert math.isclose(tcpa, 20.0, abs_tol=1e-9)


def test_dcpa_tcpa_opening_range_uses_current_distance():
    dcpa, tcpa = compute_dcpa_tcpa(0, 0, -1, 0, 10, 0, 1, 0)
    assert tcpa == 0.0 and math.isclose(dcpa, 10.0)


def test_rect_overlap_and_clearance():
    assert rect_overlap(0, 0, 6, 3, 0.0, 5, 0, 6, 3, 0.0)
    assert not rect_overlap(0, 0, 6, 3, 0.0, 10, 0, 6, 3, 0.0)
    assert min_distance_rects(0, 0, 6, 3, 0.0, 5, 0, 6, 3, 0.0) == 0.0
    # Parallel boxes with a 4 m gap between their facing edges
    assert math.isclose(min_distance_rects(0, 0, 6, 3, 0.0, 10, 0, 6, 3, 0.0), 4.0)
    # Rotation by 90 deg: half-length 3 -> half-width 1.5 along x
    assert math.isclose(min_distance_rects(0, 0, 6, 3, 0.0, 10, 0, 6, 3, math.pi / 2), 5.5)
