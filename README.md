# CC-CBF: Compliant-Course Control Barrier Function for COLREG-Aware USV Collision Avoidance

[![tests](https://github.com/Muhayyuddin/cc-cbf/actions/workflows/tests.yml/badge.svg)](https://github.com/Muhayyuddin/cc-cbf/actions/workflows/tests.yml)
[![project page](https://img.shields.io/badge/project-page-blue)](https://muhayyuddin.github.io/cc-cbf/)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **CC-CBF: Compliant-Course Control Barrier Function for USV Collision Avoidance**<br>
> Muhayy Ud Din, Waseem Akram, Ahsan Baidar Bakht, Taha Tarek, Irfan Hussain<br>
> *Submitted to IEEE Transactions on Control Systems Technology (TCST), 2026*

This repository contains the CC-CBF controller, the four baseline controllers, the simulator and the scripts that reproduce every table and simulation figure in the paper.

<p align="center">
  <img src="assets/figures/graphical_abstract.png" width="620" alt="CC-CBF under COLREG Rules 13 and 14">
</p>

---

## Overview

Two vessels approaching each other must avoid a collision, and the maritime collision regulations (COLREGs) also specify *how* each vessel must manoeuvre. In a head-on encounter both vessels turn to **starboard** and pass port to port (Rule 14). A give-way vessel in a crossing situation must not cross ahead of the other vessel (Rule 15). Conventional control barrier functions (CBFs) guarantee separation but are isotropic, so they cannot tell a lawful passing side from an unlawful one.

**CC-CBF makes the safety region directionally asymmetric.** For each target vessel, a single barrier

```
h_CC = ||p_o − p_t||² − R_CC(θ, τ)²,     R_CC = R_b · [1 + λ(τ) · Φ(θ, τ)],     Φ = [cos(θ − θ_C(τ))]₊²
```

enlarges the safety radius toward a lobe centre `θ_C` that depends on the encounter type `τ`. The lobe pushes the target's relative bearing away from the prohibited side. The CBF condition is affine in the commanded velocity, so a small quadratic program (QP) runs at 20 Hz and keeps the velocity inside the safe set. Two layers of nominal guidance choose the passing route:

* a starboard heading bias for head-on and overtaking encounters;
* a stern reference goal for crossing give-way encounters.

An anticipatory margin tightens the constraint as closing speed increases. The barrier certifies safety and the nominal layer selects the route, so the two work together (paper Sec. VII).

### Key features

- **One barrier per target** for every encounter type: the COLREG direction is encoded in `λ(τ)` and `θ_C(τ)`.
- **C¹ directional lobe** `Φ = [cos(θ − θ_C)]₊²`, plus a Rule-15 **stern-passage factor** that inflates the barrier ahead of a crossing target's bow.
- **Exact planar QP.** The decision variable is 2-D, so the QP is solved exactly by enumerating the O(n²) candidate active sets. When the problem is infeasible, the solver applies a documented constraint priority (paper Algorithm 1).
- **Real-time:** a mean of **24 µs** per control update on one CPU core, the lowest of all tested controllers.
- **Classification hysteresis** (2° dead-band) and give-way persistence until the closest point of approach (CPA), with a proven reset condition across changes of encounter type.

---

## Scenario demonstrations

### Head-on (COLREG Rule 14): both vessels alter course to starboard

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](assets/gifs/head_on_cc_cbf.gif) | ![](assets/gifs/head_on_c3bf.gif) | ![](assets/gifs/head_on_rule_colreg.gif) | ![](assets/gifs/head_on_geo_cri.gif) | ![](assets/gifs/head_on_tc_cbf.gif) |

### Crossing give-way (COLREG Rule 15): the give-way vessel passes astern

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](assets/gifs/crossing_give_way_cc_cbf.gif) | ![](assets/gifs/crossing_give_way_c3bf.gif) | ![](assets/gifs/crossing_give_way_rule_colreg.gif) | ![](assets/gifs/crossing_give_way_geo_cri.gif) | ![](assets/gifs/crossing_give_way_tc_cbf.gif) |

### Overtaking (COLREG Rule 13): the overtaking vessel keeps clear, passing the overtaken vessel on its starboard side

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](assets/gifs/overtaking_cc_cbf.gif) | ![](assets/gifs/overtaking_c3bf.gif) | ![](assets/gifs/overtaking_rule_colreg.gif) | ![](assets/gifs/overtaking_geo_cri.gif) | ![](assets/gifs/overtaking_tc_cbf.gif) |

### Static obstacle field

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](assets/gifs/static_obstacles_cc_cbf.gif) | ![](assets/gifs/static_obstacles_c3bf.gif) | ![](assets/gifs/static_obstacles_rule_colreg.gif) | ![](assets/gifs/static_obstacles_geo_cri.gif) | ![](assets/gifs/static_obstacles_tc_cbf.gif) |

### Multi-vessel encounters (CC-CBF)

| Parallel head-on (Rule 14 × 2) | Mixed rules (Rules 13 + 14) | Head-on, then overtaking |
|:------------------------------:|:---------------------------:|:------------------------:|
| ![](assets/gifs/multi_parallel_head_on.gif) | ![](assets/gifs/multi_mixed_rules.gif) | ![](assets/gifs/multi_head_on_then_overtaking.gif) |

---

## Gazebo simulation and sea trials

Click a preview to open the full video.

| | Head-on (Rule 14) | Overtaking (Rule 13) |
|:--|:--:|:--:|
| **MBZIRC Gazebo** | [![Gazebo head-on](assets/videos/gazebo_head_on_preview.gif)](assets/videos/gazebo_head_on.mp4) | [![Gazebo overtaking](assets/videos/gazebo_overtaking_preview.gif)](assets/videos/gazebo_overtaking.mp4) |
| **Sea trials** | [![Sea trial head-on](assets/videos/sea_trial_head_on_preview.gif)](assets/videos/sea_trial_head_on.mp4) | [![Sea trial overtaking](assets/videos/sea_trial_overtaking_preview.gif)](assets/videos/sea_trial_overtaking.mp4) |

All 12 sea trials in the paper were collision-free, with a minimum hull clearance of 11.4 m.

---

## Results

The numbers below are those of the paper. The [reproduction commands](#reproducing-the-paper) regenerate them, and the archived raw outputs are in [`paper_results/`](paper_results/).

**Monte Carlo** (Table VII; 50 randomised trials × 4 scenarios × 5 controllers = 1000 runs). The Sep, COL and Eff. columns are grand averages.

| Controller | Collisions | Mean Sep (m) | Min Sep (m) | COL | Eff. | Time / update |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **CC-CBF (ours)** | **0 / 200** | 18.6 | **8.6** | **1.00** | 0.96 | **24 µs** |
| C3BF | 0 / 200 | 22.5 | 12.6 | 0.40 | 0.94 | 29 µs |
| Rule-COLREG | 14 / 200 | 8.1 | 0.0 | 1.00 | 1.01 | 1.58 ms |
| Geo-CRI | 15 / 200 | 8.5 | 0.0 | 0.67 | 1.03 | 3.36 ms |
| TC-CBF | 0 / 200 | 14.2 | 8.3 | 0.79 | 0.96 | 33 µs |

Metric definitions:

* **Sep:** minimum hull-to-hull clearance. The design threshold is D_safe = 8 m.
* **COL:** per-step rule adherence during give-way encounters, averaged over the three vessel encounters.
* **Eff.:** path ratio, averaged over the trials that reached the goal.
* **Time:** mean controller computation per control update (Table III).

CC-CBF is the only controller that combines zero collisions, clearance above D_safe in every trial and full rule adherence.

**Ablation** (Table IV; N = 200 paired trials, σ_v = 1.0 m/s, σ_ψ = 10°): rate of passing on the selected side at CPA, with 0 % collisions in every configuration.

| Configuration | Head-on | Crossing | Overtaking |
|---|:--:|:--:|:--:|
| **Full CC-CBF** | **86.5 %** | **97.0 %** | **96.5 %** |
| No nominal shaping (barrier only) | 68.5 % | 35.5 % | 63.5 % |
| … and θ_C = 0 | 51.5 % | 31.5 % | 48.0 % |
| … and mirrored lobe | 35.5 % | 30.5 % | 28.0 % |
| C3BF (isotropic) | 53.5 % | 30.5 % | 49.0 % |

With nominal shaping disabled, the directional lobes alone raise the passing-side rate by 4–17 percentage points over a centred lobe (paired exact McNemar tests), and mirroring the lobes lowers it.

---

## Installation

The code runs in place from the repository root, with no package installation. It requires Python ≥ 3.9 (tested with Python 3.10, NumPy 1.26, Matplotlib 3.5).

```bash
git clone https://github.com/Muhayyuddin/cc-cbf.git
cd cc-cbf
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Quick start

The following runs the deterministic comparison of all five controllers on the four scenarios (Table III, about 1 minute):

```bash
python run_all_planners.py
python run_all_planners.py --controller CC-CBF --scenario head_on
```

Use the controller from Python:

```python
from core.scenario import get_scenario
from core.simulator import Simulator
from algorithms.cc_cbf import CCCBFController

sim = Simulator(get_scenario("head_on", seed=42), CCCBFController())
sim.run_to_completion()
m = sim.get_metrics()
print(f"min clearance {m.min_separation:.2f} m, COL {m.colreg_compliance:.2f}, "
      f"collision {m.collision_flag}")
# min clearance 19.15 m, COL 1.00, collision False
```

`sim.records` holds the full per-step log (states, targets, commands, thrusts and encounter classes). The available scenarios are `head_on`, `crossing_give_way`, `overtaking`, `static_obstacles`, `parallel_head_on`, `mixed_rules` and `head_on_then_overtaking`. Pass `speed_var` / `heading_var` to `get_scenario` for randomised trials.

---

## Reproducing the paper

Each script prints the table as reported in the paper and writes CSV files to `results/` (tables) or `figures/` (figures). Both folders are created on demand and are not tracked by git. Every script accepts `--help`.

The Monte Carlo-style scripts run in parallel on all cores by default. Use `--workers N` to limit this, and `--workers 1` for serial execution. Results are identical for any number of workers.

| Paper item | Command | Time (1 core / 30 cores) |
|---|---|:--:|
| Table III: deterministic comparison, seed 42 | `python run_all_planners.py` | 1 min |
| Table IV: component ablation, N = 200 | `python run_ablation.py` | ~50 min / 2 min |
| Tables V & VI: λ sweep, heading-noise robustness | `python run_sensitivity.py` | ~12 min / 30 s |
| Table VII: Monte Carlo, 1000 runs | `python run_mc_5way.py` | ~60 min / 2.5 min |
| Sec. VI-H: QP solver-branch statistics | `python run_solver_stats.py` | ~8 min / 20 s |
| Sec. VI-G: multi-vessel statistics | `python run_multi_vessel.py` | ~1 min / 3 s |
| Fig. 3: barrier shapes | `python generate_barrier_shape.py` | seconds |
| Fig. 5: trajectories of all controllers | `python generate_trajectories.py` | ~40 s |
| Fig. 8: h_CC traces at classification switches | `python generate_hcc_traces.py` | ~10 s |
| Fig. 9: multi-vessel scenarios (+ GIFs) | `python generate_multi_obstacle.py [--no-gif]` | 5 s (GIFs: minutes) |

Useful options:

```bash
python run_mc_5way.py --trials 10 --controllers CC-CBF C3BF   # quick subset
python run_ablation.py --variants all                         # also the extra lobe variants
python run_all_planners.py --repeats 3                        # median timing, as in the paper
python generate_hcc_traces.py --classifier memoryless         # without the 2-deg dead-band
```

Every script was re-run for this release and compared with the archived outputs in `paper_results/`. All tables and statistics reproduce as reported. Details are in [`paper_results/README.md`](paper_results/README.md). Computation times depend on the hardware. Floating-point differences between platforms can change the last digits of a few individual trials, but not the reported values.

---

## Repository structure

```
cc-cbf/
├── algorithms/
│   ├── cc_cbf.py              # CC-CBF (proposed): barrier, affine constraint, exact QP
│   ├── c3bf.py                # B1  collision-cone CBF (Tayal et al.)
│   ├── rule_based_colreg.py   # B2  rule-based COLREG + predictive filter (Benjamin et al.)
│   ├── geometric_cri.py       # B3  DCPA/TCPA collision-risk index + predictive filter
│   ├── turning_circle_cbf.py  # B4  turning-circle CBF (Lee et al.)
│   ├── cbf_filter.py          # predictive safety filter shared by B2 and B3
│   ├── base_controller.py     # controller interface and shared helpers
│   └── registry.py            # name -> controller class (paper order)
├── core/
│   ├── simulator.py           # LiDAR gating, classification, control loop, logging
│   ├── usv_model.py           # MBZIRC USV 3-DOF dynamics (twin thrusters)
│   ├── autopilot.py           # shared heading/speed autopilot with drag feed-forward
│   ├── colregs.py             # COLREG encounter classifier (with dead-band)
│   ├── scenario.py            # canonical and multi-vessel scenarios (+ perturbations)
│   ├── metrics.py             # clearance, COLREG adherence, path ratio, ...
│   ├── evaluation.py          # CPA-side scoring, statistics, parallel runner
│   ├── geometry.py            # oriented-rectangle clearance, bearings, CPA
│   └── entities.py            # data classes
├── data/config.py             # vessel, simulation and safety parameters (Table I)
├── run_*.py / generate_*.py   # experiment and figure scripts (see table above)
├── paper_results/             # archived raw outputs behind the paper
├── assets/                    # GIFs, videos and README figures
├── tests/                     # pytest suite (unit, regression and script smoke tests)
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # test dependencies
└── LICENSE                    # MIT
```

Conventions: the world frame has x east and y north, and angles are counter-clockwise, so a positive relative bearing is to port. The control and integration period is 0.05 s (20 Hz).

---

## CC-CBF parameters

<p align="center">
  <img src="assets/figures/barrier_shape.png" width="760" alt="CC-CBF barrier shapes">
</p>

| Encounter τ | λ | θ_C | Nominal shaping | COLREG rule |
|---|:--:|:--:|---|:--:|
| Head-on | 0.55 | −45° | 25° starboard heading bias | 14 |
| Crossing, give-way | 0.60 | +60° | stern reference goal (35 m astern, 65 % speed) + stern-passage factor 0.55 | 15 |
| Crossing, stand-on | 0.10 | +60° | hold course | 17 |
| Overtaking | 0.45 | −30° | 25° starboard heading bias | 13 |
| None / static | 0.00 | 0° | goal-directed | — |

The remaining parameters are shared across encounters:

* class-K gain α = 0.8 and tightening gain γ = 4.0;
* activation distance d_act = 95 m and LiDAR range 100 m;
* base radius R_b = 15.7 m (6 × 3.3 m USV, 8 × 3 m target), with buffer d_buf = D_safe = 8 m;
* cruise speed 0.75 v_max = 3.09 m/s.

The parameters are module-level constants in [`algorithms/cc_cbf.py`](algorithms/cc_cbf.py). To change them temporarily, use `with cc_cbf.overridden(gamma=0.0): ...`; this is how the ablation variants are built.

## USV model (MBZIRC)

| Parameter | Value |
|---|---|
| Hull length × beam | 6.0 × 3.3 m |
| Mass / yaw inertia | 200 kg / 200 kg·m² |
| Linear / quadratic surge damping | 51.3 N·s/m / 72.4 N·s²/m² |
| Yaw damping | 400 N·m·s/rad |
| Thruster moment arm | 1.348 m |
| Maximum speed | 4.12 m/s (8 kn) |
| Yaw-rate saturation (simulator) | 1.0 rad/s |

## Baselines

All baselines share the USV dynamics, autopilot, LiDAR model, simulator and goal-reaching logic.

| ID | Method | Reference | Description |
|---|---|---|---|
| B1 | **C3BF** | Tayal et al., ACC 2024 | Isotropic collision-cone CBF, no COLREG awareness |
| B2 | **Rule-COLREG** | Benjamin et al., ICRA 2006 | Deterministic rule-based manoeuvres + predictive safety filter |
| B3 | **Geo-CRI** | Huang et al., Safety Sci. 2020 | DCPA/TCPA risk index, threshold-triggered avoidance + predictive safety filter |
| B4 | **TC-CBF** | Lee et al., Mechatronics 2026 | Turning-circle CBF with encounter-dependent port/starboard barriers |

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # unit, regression and smoke tests (~1 min)
pytest -m slow              # remaining Table III regression runs and Fig. 5
```

The suite checks the following:

* the scalar fast path against the NumPy reference implementation;
* optimality of the exact QP;
* equivalence of the two COLREG classifiers;
* the statistics helpers against SciPy;
* the Table III values of the paper;
* that every experiment script runs end to end.

---

## Citation

If you use this code, please cite the paper. The entry will be updated on publication.

```bibtex
@article{uddin2026cccbf,
  title   = {{CC-CBF}: Compliant-Course Control Barrier Function for {USV} Collision Avoidance},
  author  = {Ud Din, Muhayy and Akram, Waseem and Bakht, Ahsan Baidar and Tarek, Taha and Hussain, Irfan},
  journal = {IEEE Transactions on Control Systems Technology},
  note    = {Under review},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This work was supported by Khalifa University under Award Nos. RC1-2018-KUCARS-8474000136, MBZIRC-8434000194 and KU-DFL-8475000016. The USV parameters come from the [MBZIRC simulator](https://github.com/osrf/mbzirc) model.
