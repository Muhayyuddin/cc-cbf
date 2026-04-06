# CC-CBF: Compliant-Course Control Barrier Function for COLREG-Aware USV Collision Avoidance

> **"CC-CBF: Compliant-Course Control Barrier Function for USV Collision Avoidance"**  
> *IEEE Transactions on Control Systems Technology (TCST), 2026*

---

## What does this do? (plain English)

Imagine two ships approaching each other at sea. Maritime law (COLREG) has strict rules about *which side* each vessel must manoeuvre to — for example, in a head-on situation both ships must turn **to starboard (right)**, never to port. Most autonomous collision avoidance systems either just try to avoid a collision *somehow* (without caring about direction), or follow rigid hand-coded rules that can fail in ambiguous or multi-vessel situations.

**CC-CBF takes a different approach.** It shapes the "danger zone" around each obstacle to be *directionally asymmetric* — the forbidden region is made larger on the side that COLREG prohibits, and narrower on the compliant side. This is encoded mathematically as a single barrier function, and a small optimisation (QP) runs at 20 Hz to continuously keep the vessel's velocity on the legal, safe side. The result: the ship **automatically turns the right way** as a direct consequence of the geometry — no rule lookup table, no mode switching, no separate safety filter needed.

The key insight is that by inflating the barrier radius on the prohibited side, the QP's minimum-effort solution naturally steers toward compliance. If the obstacle is to the port-bow in a head-on encounter, the barrier is bigger there, so the solver deflects to starboard — exactly what COLREG Rule 14 demands — without ever explicitly "knowing" the rule.

---

## Scenario Demonstrations

### Head-On (COLREG Rule 14) — both vessels turn to starboard

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](gifs_paper/head_on_cc_cbf.gif) | ![](gifs_paper/head_on_c3bf.gif) | ![](gifs_paper/head_on_rule_colreg.gif) | ![](gifs_paper/head_on_geo_cri.gif) | ![](gifs_paper/head_on_tc_cbf.gif) |
| ✅ COLREG 1.00 | ⚠️ COLREG 0.78 | ⚠️ COLREG 0.91 | ⚠️ COLREG 0.74 | ✅ COLREG 1.00 |

### Crossing Give-Way (COLREG Rule 15) — give-way vessel passes astern

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](gifs_paper/crossing_give_way_cc_cbf.gif) | ![](gifs_paper/crossing_give_way_c3bf.gif) | ![](gifs_paper/crossing_give_way_rule_colreg.gif) | ![](gifs_paper/crossing_give_way_geo_cri.gif) | ![](gifs_paper/crossing_give_way_tc_cbf.gif) |
| ✅ COLREG 0.96 | ❌ COLREG 0.31 | ✅ COLREG 0.93 | ✅ COLREG 0.98 | ⚠️ COLREG 0.92 |

### Overtaking (COLREG Rule 13) — overtaking vessel passes to starboard

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](gifs_paper/overtaking_cc_cbf.gif) | ![](gifs_paper/overtaking_c3bf.gif) | ![](gifs_paper/overtaking_rule_colreg.gif) | ![](gifs_paper/overtaking_geo_cri.gif) | ![](gifs_paper/overtaking_tc_cbf.gif) |
| ✅ COLREG 1.00 | ❌ COLREG 0.00 | ✅ COLREG 1.00 | ❌ COLREG 0.38 | ✅ COLREG 1.00 |

### Static Obstacle Field — multi-obstacle avoidance

| **CC-CBF (ours)** | C3BF | Rule-COLREG | Geo-CRI | TC-CBF |
|:-----------------:|:----:|:-----------:|:-------:|:------:|
| ![](gifs_paper/static_obstacles_cc_cbf.gif) | ![](gifs_paper/static_obstacles_c3bf.gif) | ![](gifs_paper/static_obstacles_rule_colreg.gif) | ![](gifs_paper/static_obstacles_geo_cri.gif) | ![](gifs_paper/static_obstacles_tc_cbf.gif) |
| ✅ Sep 12.9 m | ✅ Sep 10.9 m | ✅ Sep 3.8 m | ⚠️ Sep 5.1 m | ⚠️ Sep 5.2 m |

---

## Overview

**CC-CBF** encodes both **safety** (minimum separation) and **COLREG compliance** (maritime right-of-way rules) in a single directionally asymmetric barrier function per obstacle. The encounter type (head-on, crossing, overtaking) is embedded directly into the barrier radius via cosine modulation, so the QP always projects the nominal velocity onto the COLREG-compliant side — no external rule-switching logic required.

### Key Features
- **Single barrier function** per obstacle — no hybrid automaton, no Zeno-behaviour concerns
- **Directional asymmetry** `R_CC = R_base · (1 + λ · Φ(θ, τ))` encodes which side to pass on
- **Velocity-dependent tightening** anticipates close encounters at high speed
- **Stern-waypoint nominal** for crossing give-way: points 35 m behind the target transom + 35% speed reduction
- **O(n) projection solver** — <0.5 ms per step, well within the 50 ms @ 20 Hz budget

---

## Monte Carlo Results (50 trials × 4 scenarios × 5 controllers)

| Controller        | Coll % | Min Sep (m) | COLREG | Efficiency |
|-------------------|:------:|:-----------:|:------:|:----------:|
| **CC-CBF (Ours)** | **0.0**| **13.0**    |**0.99**| 0.98       |
| C3BF              | 0.0    | 18.5        | 0.38   | 0.94       |
| Rule-COLREG       | 2.5    | 9.9         | 0.96   | **1.00**   |
| Geo-CRI           | 9.0    | 8.4         | 0.65   | 1.02       |
| TC-CBF            | 1.0    | 10.5        | 0.87   | 0.86       |

CC-CBF is the **only controller with 0 % collisions** across all trials and achieves the highest COLREG compliance average.

---

## Project Structure

```
collision-avoidance/
│
├── algorithms/                     # Controller implementations
│   ├── cc_cbf.py                   # ← CC-CBF (proposed method)
│   ├── c3bf.py                     # Baseline B1: Collision-Cone CBF
│   ├── rule_based_colreg.py        # Baseline B2: Rule-based COLREG
│   ├── geometric_cri.py            # Baseline B3: Geometric CRI
│   ├── turning_circle_cbf.py       # Baseline B4: Turning-Circle CBF
│   └── base_controller.py          # Abstract base class
│
├── core/                           # Simulation engine
│   ├── simulator.py                # Main simulation loop
│   ├── scenario.py                 # Scenario generators (+ MC perturbation)
│   ├── colregs.py                  # COLREG encounter classifier
│   ├── metrics.py                  # MinSep, compliance, efficiency
│   ├── entities.py                 # USVState, Obstacle, StepRecord
│   ├── usv_model.py                # MBZIRC-class USV dynamics
│   ├── autopilot.py                # Inner-loop heading/speed controller
│   └── geometry.py                 # Collision geometry utilities
│
├── data/
│   └── config.py                   # Simulation configuration constants
│
├── paper/
│   ├── short-version.tex           # ← Main paper source (TCST submission)
│   ├── short-version.pdf           # Compiled PDF
│   ├── IEEEtran.bst                # BibTeX style
│   ├── references.bib              # Bibliography
│   └── figures/                    # All paper figures (PNG + EPS)
│
├── gifs_paper/                     # Animated GIFs (5 controllers × 4 scenarios)
│
├── run_mc_5way.py                  # 5-way Monte Carlo (50 trials × 4 scenarios)
├── run_ablation.py                 # Component ablation study (8 CC-CBF variants)
├── run_sensitivity.py              # λ-sensitivity + noise-robustness sweep
├── generate_multi_obstacle.py      # Multi-vessel stress scenario plots (PNG + EPS)
├── generate_paper_gifs.py          # Animated GIFs for all controllers/scenarios
├── generate_graphical_abstract.py  # Graphical abstract figure
├── draw_architecture.py            # Three-layer architecture block diagram
│
├── results_mc_5way.csv             # Pre-computed MC results (1 000 runs)
├── results_ablation.csv            # Pre-computed ablation results
├── results_lambda_sweep.csv        # Pre-computed λ-sweep results
├── results_noise_robustness.csv    # Pre-computed noise-robustness results
├── results_sensitivity.csv         # Pre-computed sensitivity results
│
└── requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd collision-avoidance
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick sanity check

Run all five planners across all four scenarios in ~10 seconds (single deterministic seed, no randomisation):

```bash
python run_all_planners.py
# Optionally filter:
python run_all_planners.py --controller CC-CBF
python run_all_planners.py --scenario head_on
```

---

## Reproducing Paper Results

### 1 — Monte Carlo evaluation (5-way, 50 trials)
```bash
python run_mc_5way.py --trials 50
# Output: results_mc_5way.csv  (~70 min on a laptop)
```

### 2 — Component ablation study
```bash
python run_ablation.py
# Output: results_ablation.csv  (~10 min)
```

### 3 — λ-sensitivity and noise-robustness study
```bash
python run_sensitivity.py
# Output: results_lambda_sweep.csv, results_noise_robustness.csv  (~4 min)
```

### 4 — Multi-vessel stress scenario figures
```bash
python generate_multi_obstacle.py --no-gif
# Output: paper/figures/multi_obstacle_parallel_head_on.{png,eps}
#         paper/figures/multi_obstacle_mixed_rules.{png,eps}
```

### 5 — Animated GIFs (all 20 controller × scenario combinations)
```bash
python generate_paper_gifs.py
# Output: gifs_paper/*.gif

# Specific controller or scenario:
python generate_paper_gifs.py --controller CC-CBF --scenario head_on
```
Available controllers: `CC-CBF`, `C3BF`, `Rule-COLREG`, `Geo-CRI`, `TC-CBF`  
Available scenarios: `head_on`, `crossing_give_way`, `overtaking`, `static_obstacles`

### 6 — Graphical abstract and architecture figures
```bash
python generate_graphical_abstract.py
python draw_architecture.py
```

### 7 — Compile the paper
```bash
cd paper
pdflatex short-version.tex
bibtex short-version
pdflatex short-version.tex
pdflatex short-version.tex
```

---

## CC-CBF Parameters

| Encounter   | λ    | θ_C  | Nominal Bias                    | COLREG Rule |
|-------------|:----:|:----:|---------------------------------|:-----------:|
| Head-on     | 0.55 | −45° | 25° starboard heading bias      | 14          |
| Crossing GW | 0.60 | +60° | Stern-waypoint + 35 % speed cut | 15          |
| Crossing SO | 0.10 | +60° | Hold course                     | 17          |
| Overtaking  | 0.45 | +30° | 25° starboard heading bias      | 13          |
| None        | 0.00 |  0°  | Hold course                     | —           |

**Shared parameters:**  
`α = 0.8` (CBF class-K gain) · `γ = 4.0` (tightening gain) · `d_act = 95 m` (activation range)  
`R_base = 12.7 m` · `D_safe = 10.0 m` · `LIDAR_RANGE = 100 m`

---

## USV Model (MBZIRC-class)

| Parameter               | Value               |
|-------------------------|---------------------|
| Hull length × beam      | 6.0 × 3.3 m         |
| Mass                    | 200 kg              |
| Yaw inertia             | 200 kg·m²           |
| Linear surge damping    | 51.3 N·s/m          |
| Quadratic surge damping | 72.4 N·s²/m²        |
| Yaw damping             | 400 N·m·s/rad       |
| Thruster moment arm     | 1.348 m             |
| Maximum speed           | 4.12 m/s (~8 knots) |

---

## Baselines

| ID | Method          | Reference             | Description                                               |
|----|-----------------|-----------------------|-----------------------------------------------------------|
| B1 | **C3BF**        | Tayal et al. 2024     | Collision-Cone CBF, isotropic — no COLREG awareness       |
| B2 | **Rule-COLREG** | Benjamin et al. 2006  | Deterministic rule-based manoeuvres + distance-CBF filter |
| B3 | **Geo-CRI**     | Huang et al. 2020     | DCPA/TCPA risk index, threshold-triggered avoidance       |
| B4 | **TC-CBF**      | Lee et al. 2025       | Turning-circle CBF with port/starboard barrier pairs      |

All baselines share the same USV dynamics, autopilot, simulation engine, and goal-reaching logic for a fair comparison.

---

## Citation

```bibtex
@article{ccbf_tcst2026,
  title   = {{CC-CBF}: Compliant-Course Control Barrier Function
             for {USV} Collision Avoidance},
  author  = {Din, Muhayy Ud and Akram, Waseem and Bakht, Ahsan B. and others},
  journal = {IEEE Transactions on Control Systems Technology},
  year    = {2026},
}
```
