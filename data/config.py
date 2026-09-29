"""
Simulation and vessel configuration shared by every controller.

MBZIRC USV parameters are taken from the MBZIRC simulator model
(https://github.com/osrf/mbzirc, ``mbzirc_ign/models/usv/model.sdf.erb``):

* ``[MBZIRC]`` values come directly from the SDF;
* ``[APPROX]`` values are approximations for a 6 m class USV.

The values reproduce Table I of the paper.  Controller-specific parameters
(Table II) live next to each controller in ``algorithms/``.
"""

import math

# ---------------------------------------------------------------------------
# USV physical parameters (paper Table I)
# ---------------------------------------------------------------------------

# [MBZIRC] Hull geometry (Surface plugin)
USV_LENGTH = 6.0          # m, vehicle_length
USV_WIDTH = 3.3           # m, vehicle_width

# [MBZIRC] Hydrodynamic drag (SimpleHydrodynamics plugin)
X_U = 51.3                # N/(m/s), linear surge drag
X_UU = 72.4               # N/(m/s)^2, quadratic surge drag
N_R = 400.0               # N*m/(rad/s), yaw drag (nR in the SDF)

# [MBZIRC] Speed limit
MAX_SPEED_KNOTS = 8.0
MAX_SPEED_MPS = MAX_SPEED_KNOTS * 0.5144  # 4.1152 m/s

# [MBZIRC] Thrust limits (computed in the SDF ERB template)
MAX_TOTAL_THRUST = (X_U + X_UU * MAX_SPEED_MPS) * MAX_SPEED_MPS  # ~1437.19 N
MAX_THRUSTER_THRUST = MAX_TOTAL_THRUST / 2.0                      # ~718.60 N

# [APPROX] Mass and inertia (not in the SDF)
USV_MASS = 200.0            # kg, typical for a 6 m class USV
USV_YAW_INERTIA = 200.0     # kg*m^2, approx. m * (L/4)^2
USV_YAW_DAMPING = N_R       # N*m*s/rad
THRUSTER_LEVER_ARM = 1.348  # m, lateral thruster offset [MBZIRC SDF: y = +-1.348]
MAX_YAW_RATE = 1.0          # rad/s, yaw-rate saturation of the dynamics model
MIN_SURGE_SPEED = -0.5      # m/s, small reverse speed allowed when braking
MAX_REVERSE_THRUST_FRACTION = 0.1  # reverse thrust limit as a fraction of max

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIM_DT = 0.05             # s, integration and control period (20 Hz)
CRUISE_SPEED = MAX_SPEED_MPS * 0.75  # m/s, nominal speed of every controller
SIM_DURATION = 180.0      # s, time limit per run
GOAL_RADIUS = 10.0        # m, a run ends once the USV is inside this disc

# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------
# The MBZIRC planar LiDAR has a 30 m range; for maritime collision avoidance
# a representative 3-D LiDAR range of ~100 m on small vessels is used
# (cf. Zantopp et al. 2024, Yu et al. 2025).
LIDAR_RANGE = 100.0       # m, obstacles beyond this range are not reported

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
SAFETY_BUFFER = 8.0       # m, d_buf: h >= 0 implies hull clearance >= D_SAFE
D_SAFE = 8.0              # m, hull-clearance threshold (also used by Rule/CRI risk terms)
AVOIDANCE_DOMAIN = 30.0   # m, "course holding is adherent" beyond this clearance

# ---------------------------------------------------------------------------
# Shared autopilot gains [APPROX] (identical for every controller)
# ---------------------------------------------------------------------------
KP_SPEED = 150.0          # speed proportional gain
KI_SPEED = 10.0           # speed integral gain
KP_HEADING = 4.0          # heading proportional gain
KD_HEADING = 1.5          # heading derivative (yaw-rate) damping
K_YAW_TO_THRUST = 300.0   # yaw command -> differential thrust

# ---------------------------------------------------------------------------
# Baseline parameters
# ---------------------------------------------------------------------------

# Geo-CRI (B3)
GEO_CRI_THRESHOLD = 0.6                 # risk threshold that triggers avoidance
GEO_CRI_STBD_TURN = math.radians(30)    # starboard turn magnitude
GEO_CRI_SLOW_FACTOR = 0.5               # speed reduction factor

# Predictive safety filter shared by Rule-COLREG (B2) and Geo-CRI (B3)
CBF_HORIZON = 3.0          # s, prediction horizon
CBF_STEPS = 6              # prediction steps over the horizon
CBF_MIN_SEPARATION = 8.0   # m, minimum admissible predicted separation

# ---------------------------------------------------------------------------
# Target defaults
# ---------------------------------------------------------------------------
DEFAULT_TARGET_LENGTH = 8.0   # m
DEFAULT_TARGET_WIDTH = 3.0    # m

# ---------------------------------------------------------------------------
# Monte Carlo perturbations (paper Sec. VI-A)
# ---------------------------------------------------------------------------
MONTE_CARLO_LATERAL_OFFSET_STD = 5.0        # m, head-on lateral offset
MONTE_CARLO_SPEED_STD = 0.3                 # m/s
MONTE_CARLO_HEADING_STD = math.radians(5)   # rad
