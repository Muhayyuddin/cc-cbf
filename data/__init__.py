"""
Configuration defaults for the LC-CRI collision avoidance simulator.

MBZIRC USV parameters are sourced from:
  https://github.com/osrf/mbzirc  (mbzirc_ign/models/usv/model.sdf.erb)

Parameters marked [MBZIRC] come directly from the SDF.
Parameters marked [APPROX] are reasonable approximations for a 6 m class USV.
"""

from dataclasses import dataclass, field
from typing import Dict
import math

# ---------------------------------------------------------------------------
# MBZIRC USV Physical Parameters
# ---------------------------------------------------------------------------

# [MBZIRC] Hull geometry from Surface plugin
USV_LENGTH = 6.0          # m, vehicle_length
USV_WIDTH = 3.3           # m, vehicle_width
USV_BODY_HEIGHT = 0.27    # m, hull_radius (for metadata)

# [MBZIRC] Hydrodynamic drag from SimpleHydrodynamics plugin
X_U = 51.3                # N/(m/s), linear surge drag
X_UU = 72.4               # N/(m/s)^2, quadratic surge drag
Y_V = 40.0                # N/(m/s), linear sway drag (from yV in SDF)
N_R = 400.0               # N·m/(rad/s), yaw drag (from nR in SDF)

# [MBZIRC] Speed limits
MAX_SPEED_KNOTS = 8.0
MAX_SPEED_MPS = MAX_SPEED_KNOTS * 0.5144  # 4.1152 m/s

# [MBZIRC] Thrust limits (computed in SDF ERB)
MAX_TOTAL_THRUST = (X_U + X_UU * MAX_SPEED_MPS) * MAX_SPEED_MPS  # ~1437.19 N
MAX_THRUSTER_THRUST = MAX_TOTAL_THRUST / 2.0  # ~718.60 N

# [APPROX] Mass and inertia - not in SDF snippet
USV_MASS = 200.0          # kg, typical for 6 m class USV
USV_YAW_INERTIA = 200.0   # kg·m², approx m*(L/4)^2
USV_YAW_DAMPING = 400.0   # N·m·s/rad, using nR from SDF
THRUSTER_LEVER_ARM = 1.348 # m, lateral offset from centreline [MBZIRC SDF: y=±1.348]

# ---------------------------------------------------------------------------
# Simulation Parameters
# ---------------------------------------------------------------------------
SIM_DT = 0.05             # s, simulation time step
SIM_DURATION = 180.0      # s, max simulation duration (200 m at ~3 m/s ≈ 67 s + manoeuvre margin)
RENDER_FPS = 30           # GUI update rate

# ---------------------------------------------------------------------------
# Sensor Parameters (MBZIRC USV LiDAR)
# ---------------------------------------------------------------------------
# [MBZIRC] mbzirc_planar_lidar: gpu_ray, FOV ±135°, 720 samples, 30 Hz
#   SDF max_range = 30 m (short-range 2D laser scanner)
# For realistic maritime CA we use a representative range (e.g. Velodyne VLP-16
# class, ~100 m effective detection on small vessels at sea — consistent with
# Zantopp et al. 2024 "up to 150 m", Yu et al. 2025 "100 m").
LIDAR_RANGE = 100.0       # m, effective obstacle detection range

# ---------------------------------------------------------------------------
# Safety Parameters
# ---------------------------------------------------------------------------
SAFETY_BUFFER = 5.0       # m, inflated footprint buffer around hull
AVOIDANCE_DOMAIN = 30.0   # m, planning avoidance radius
D_SAFE = 10.0             # m, minimum safe separation (center-to-center)
D_EMERGENCY = 5.0         # m, emergency threshold (collision-like)

# ---------------------------------------------------------------------------
# Autopilot Gains [APPROX] - tuned for stable MBZIRC-class USV behavior
# ---------------------------------------------------------------------------
KP_SPEED = 150.0          # speed proportional gain
KI_SPEED = 10.0           # speed integral gain (small)
KP_HEADING = 200.0        # heading proportional gain
KD_HEADING = 80.0         # heading derivative (yaw rate) damping
K_YAW_TO_THRUST = 300.0   # yaw command to differential thrust mapping

# ---------------------------------------------------------------------------
# Algorithm Parameters
# ---------------------------------------------------------------------------

# Geometric CRI
GEO_CRI_THRESHOLD = 0.6   # risk threshold to trigger avoidance
GEO_CRI_STBD_TURN = math.radians(30)  # starboard turn magnitude
GEO_CRI_SLOW_FACTOR = 0.5  # speed reduction factor

# Rule-based COLREG
COLREG_BEARING_TOLERANCE = math.radians(15)  # tolerance for head-on classification
COLREG_OVERTAKING_SECTOR = math.radians(67.5)  # sector behind target

# LC-CRI
LCRI_RISK_WEIGHTS = {
    'Rg': 0.25,  # geometric
    'Rd': 0.25,  # dynamic
    'Rr': 0.20,  # regulatory
    'Rs': 0.15,  # semantic
    'Ri': 0.15,  # intent
}
LCRI_TRIGGER_THRESHOLD = 0.15  # risk divergence to trigger LLM query
LCRI_MIN_REQUERY_INTERVAL = 3.0  # s, minimum time between queries

# Velocity Obstacle
VO_TIME_HORIZON = 20.0     # s, VO prediction time horizon (generous lookahead)
VO_SAFETY_BUFFER = 5.0     # m, additional buffer beyond hull extents

# CBF Safety Filter
CBF_HORIZON = 3.0          # s, prediction horizon (increased for better lookahead)
CBF_STEPS = 10             # number of prediction steps
CBF_MIN_SEPARATION = 8.0   # m, minimum allowed predicted separation

# ---------------------------------------------------------------------------
# Target / Obstacle Defaults
# ---------------------------------------------------------------------------
DEFAULT_TARGET_LENGTH = 8.0   # m
DEFAULT_TARGET_WIDTH = 3.0    # m
DEFAULT_TARGET_SPEED = 2.0    # m/s

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
MONTE_CARLO_DEFAULT_TRIALS = 20
MONTE_CARLO_LATERAL_OFFSET_STD = 5.0  # m
MONTE_CARLO_SPEED_STD = 0.3           # m/s
MONTE_CARLO_HEADING_STD = math.radians(5)  # rad

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
CANVAS_SIZE_M = 400.0     # m, total canvas side length
PIXELS_PER_METER = 2.5    # rendering scale (adjusted for 400 m canvas)

# Colors (R, G, B, A) normalized 0-1 for matplotlib / Qt
COLOR_OWNSHIP = (0.0, 0.4, 0.8, 1.0)       # blue
COLOR_TARGET = (0.85, 0.2, 0.15, 1.0)       # red
COLOR_STATIC_OBS = (0.5, 0.5, 0.5, 1.0)     # gray
COLOR_SAFETY = (0.0, 0.8, 0.0, 0.3)         # green translucent
COLOR_TRAJECTORY = (0.0, 0.4, 0.8, 0.5)     # blue translucent
COLOR_TARGET_TRAJ = (0.85, 0.2, 0.15, 0.5)  # red translucent
