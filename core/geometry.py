"""
Geometry utilities for oriented rectangle collision detection,
distance computation, and footprint operations.
"""

import math
import numpy as np
from typing import Tuple, List


def rotation_matrix(angle: float) -> np.ndarray:
    """2D rotation matrix for given angle in radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def oriented_rect_corners(cx: float, cy: float, length: float,
                          width: float, heading: float) -> np.ndarray:
    """
    Compute 4 corners of an oriented rectangle.

    Args:
        cx, cy: center position
        length: along heading direction
        width: perpendicular to heading
        heading: radians

    Returns:
        (4, 2) array of corner positions
    """
    R = rotation_matrix(heading)
    hl, hw = length / 2.0, width / 2.0
    # Corners in body frame: front-right, front-left, back-left, back-right
    local = np.array([
        [hl, -hw],
        [hl,  hw],
        [-hl,  hw],
        [-hl, -hw],
    ])
    world = local @ R.T + np.array([cx, cy])
    return world


def project_polygon(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project polygon corners onto an axis, return (min, max) projection."""
    projections = corners @ axis
    return float(projections.min()), float(projections.max())


def sat_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """
    Separating Axis Theorem overlap test for two convex polygons.

    Returns True if the polygons overlap (collision).
    """
    for corners in [corners_a, corners_b]:
        n = len(corners)
        for i in range(n):
            edge = corners[(i + 1) % n] - corners[i]
            # Normal to edge
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis = axis / norm

            min_a, max_a = project_polygon(corners_a, axis)
            min_b, max_b = project_polygon(corners_b, axis)

            if max_a < min_b or max_b < min_a:
                return False  # Separating axis found
    return True  # No separating axis => overlap


def rect_overlap(cx1, cy1, l1, w1, h1, cx2, cy2, l2, w2, h2) -> bool:
    """Check if two oriented rectangles overlap."""
    c1 = oriented_rect_corners(cx1, cy1, l1, w1, h1)
    c2 = oriented_rect_corners(cx2, cy2, l2, w2, h2)
    return sat_overlap(c1, c2)


def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to line segment a-b."""
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, np.dot(p - a, ab) / ab_sq))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def min_distance_rects(cx1, cy1, l1, w1, h1, cx2, cy2, l2, w2, h2) -> float:
    """
    Approximate minimum distance between two oriented rectangles.

    Uses vertex-to-edge distances. Returns 0 if overlapping.
    """
    c1 = oriented_rect_corners(cx1, cy1, l1, w1, h1)
    c2 = oriented_rect_corners(cx2, cy2, l2, w2, h2)

    if sat_overlap(c1, c2):
        return 0.0

    min_d = float('inf')
    # Check each vertex of rect1 against edges of rect2 and vice versa
    for corners_check, corners_edge in [(c1, c2), (c2, c1)]:
        n = len(corners_edge)
        for p in corners_check:
            for i in range(n):
                a = corners_edge[i]
                b = corners_edge[(i + 1) % n]
                d = point_to_segment_dist(p, a, b)
                if d < min_d:
                    min_d = d

    return min_d


def center_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def bearing_from_to(x1: float, y1: float, psi1: float,
                    x2: float, y2: float) -> float:
    """
    Relative bearing from vessel at (x1,y1) heading psi1 to point (x2,y2).

    Returns angle in [-pi, pi]. Positive = starboard, negative = port.
    """
    dx = x2 - x1
    dy = y2 - y1
    absolute_bearing = math.atan2(dy, dx)
    return wrap_angle(absolute_bearing - psi1)


def compute_dcpa_tcpa(x1: float, y1: float, vx1: float, vy1: float,
                      x2: float, y2: float, vx2: float, vy2: float) -> Tuple[float, float]:
    """
    Compute Distance at Closest Point of Approach (DCPA) and
    Time to CPA (TCPA) for two objects moving at constant velocity.

    Returns:
        (dcpa, tcpa) in (meters, seconds)
    """
    # Relative position and velocity
    dx = x2 - x1
    dy = y2 - y1
    dvx = vx2 - vx1
    dvy = vy2 - vy1

    vrel_sq = dvx * dvx + dvy * dvy

    if vrel_sq < 1e-8:
        # Essentially same velocity -> CPA is at current position
        dcpa = math.sqrt(dx * dx + dy * dy)
        tcpa = float('inf')
        return dcpa, tcpa

    # Time of CPA
    tcpa = -(dx * dvx + dy * dvy) / vrel_sq

    if tcpa < 0:
        # CPA is in the past; use current distance
        tcpa = 0.0

    # Position at CPA
    cpx = dx + dvx * tcpa
    cpy = dy + dvy * tcpa
    dcpa = math.sqrt(cpx * cpx + cpy * cpy)

    return dcpa, tcpa
