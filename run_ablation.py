#!/usr/bin/env python3
"""
Component ablation study for CC-CBF paper.

Eight variants tested across all 3 COLREG scenarios, 40 seeds each:

  1. CC-CBF (Full)                -- all components active (proposed method)
  2. No direction                 -- THETA_C=0 for all (Layer 2 off)
  3. No stern-pass                -- STERN_PASS_LAMBDA=0 (Layer 3 off)
  4. No anticipation              -- GAMMA=0 (anticipatory tightening off)
  5. C3BF                         -- isotropic baseline (lambda=0 everywhere)
  6. No direction + No stern-pass -- Layers 2 & 3 both off
  7. No direction + No anticipation  -- Layer 2 off + anticipation off
  8. No stern-pass + No anticipation -- Layer 3 off + anticipation off

Metrics: Side% (COLREG side at CPA), Sep (min separation), Coll% (collisions)
"""

import os, sys, csv, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario  import get_scenario
from core.simulator import Simulator
from core.entities  import USVParameters
from core.geometry  import wrap_angle
from algorithms.cc_cbf import CCCBFController, _phi_and_dphi, _obs_rb
import algorithms.cc_cbf as ccbf_mod
from algorithms.c3bf import C3BFController

SCENARIOS   = ["head_on", "crossing_give_way", "overtaking"]
N_TRIALS    = 40
SEED_OFFSET = 300
SPEED_VAR   = 1.0
HEADING_VAR = math.radians(10)

CSV_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_ablation.csv")

_ORIG_LAMBDA       = dict(ccbf_mod.LAMBDA)
_ORIG_THETA_C      = dict(ccbf_mod.THETA_C)
_ORIG_GAMMA        = ccbf_mod.GAMMA
_ORIG_STERN        = ccbf_mod.STERN_PASS_LAMBDA

def _restore():
    for k in ccbf_mod.LAMBDA:    ccbf_mod.LAMBDA[k]  = _ORIG_LAMBDA[k]
    for k in ccbf_mod.THETA_C:   ccbf_mod.THETA_C[k] = _ORIG_THETA_C[k]
    ccbf_mod.GAMMA             = _ORIG_GAMMA
    ccbf_mod.STERN_PASS_LAMBDA = _ORIG_STERN

def _correct_side(rec, scn):
    if rec.state is None or not rec.obstacles: return True
    own, obs = rec.state, rec.obstacles[0]
    dx, dy = obs.x - own.x, obs.y - own.y
    if scn in ("head_on", "overtaking"):
        return wrap_angle(math.atan2(dy, dx) - own.psi) > 0.0
    return ((own.x-obs.x)*math.cos(obs.psi) + (own.y-obs.y)*math.sin(obs.psi)) < 0.0

def _run(scn, seed, variant):
    """Run one trial with the given variant configuration."""
    try:
        if variant == "full":
            pass  # defaults
        elif variant == "no_direction":
            for k in ccbf_mod.THETA_C: ccbf_mod.THETA_C[k] = 0.0
        elif variant == "no_sternpass":
            ccbf_mod.STERN_PASS_LAMBDA = 0.0
        elif variant == "no_anticipation":
            ccbf_mod.GAMMA = 0.0
        elif variant == "c3bf":
            pass  # handled below
        # --- pairwise combinations ---
        elif variant == "no_dir_no_stern":
            for k in ccbf_mod.THETA_C: ccbf_mod.THETA_C[k] = 0.0
            ccbf_mod.STERN_PASS_LAMBDA = 0.0
        elif variant == "no_dir_no_anticipation":
            for k in ccbf_mod.THETA_C: ccbf_mod.THETA_C[k] = 0.0
            ccbf_mod.GAMMA = 0.0
        elif variant == "no_stern_no_anticipation":
            ccbf_mod.STERN_PASS_LAMBDA = 0.0
            ccbf_mod.GAMMA = 0.0

        sc = get_scenario(scn, seed=seed, speed_var=SPEED_VAR, heading_var=HEADING_VAR)
        if variant == "c3bf":
            ctrl = C3BFController()
        else:
            ctrl = CCCBFController()
        sim = Simulator(sc, ctrl, USVParameters())
        sim.run_to_completion()
        m   = sim.get_metrics()
        cpa = min(sim.records, key=lambda r: r.min_distance)
        side = _correct_side(cpa, scn)
        commit = m.manoeuvre_commit_time if m.manoeuvre_commit_time < 1e6 else 0.0
        return dict(sep=m.min_separation, side=int(side),
                    commit=commit, coll=int(m.collision_flag))
    finally:
        _restore()

def _agg(rs):
    n = len(rs)
    return dict(
        sep      = round(sum(r["sep"]  for r in rs)/n, 2),
        side_pct = round(100*sum(r["side"] for r in rs)/n, 1),
        commit   = round(sum(r["commit"] for r in rs)/n, 1),
        coll_pct = round(100*sum(r["coll"] for r in rs)/n, 1),
    )

VARIANTS = [
    ("full",                     "CC-CBF (Full)"),
    ("no_direction",             "No direction (THETA\\_C=0)"),
    ("no_sternpass",             "No stern-pass"),
    ("no_anticipation",          "No anticipation (GAMMA=0)"),
    ("c3bf",                     "C3BF (isotropic)"),
    ("no_dir_no_stern",          "No dir + No stern"),
    ("no_dir_no_anticipation",   "No dir + No anticipation"),
    ("no_stern_no_anticipation", "No stern + No anticipation"),
]

def main():
    t0 = time.time()
    total = len(SCENARIOS) * len(VARIANTS) * N_TRIALS
    done  = 0

    print("CC-CBF Component Ablation Study")
    print(f"  Scenarios={len(SCENARIOS)}  Variants={len(VARIANTS)}  N={N_TRIALS}  Total={total} runs")
    print(f"  speed_var={SPEED_VAR}m/s  heading_var=10deg\n")

    hdr = f"  {'Variant':<30}{'Sep(m)':>7}  {'Side%':>6}  {'Coll%':>6}  {'Commit':>7}"
    rows = []

    for scn in SCENARIOS:
        print(f"{'='*65}")
        print(f"  Scenario: {scn}")
        print(f"{'='*65}")
        print(hdr)
        print("  " + "-"*58)

        for vkey, vlabel in VARIANTS:
            rs = [_run(scn, SEED_OFFSET+i, vkey) for i in range(N_TRIALS)]
            done += N_TRIALS
            a = _agg(rs)
            elapsed = time.time()-t0
            eta = elapsed/done*(total-done)
            print(f"  {vlabel:<30}{a['sep']:>7.2f}  {a['side_pct']:>5.1f}%  "
                  f"{a['coll_pct']:>5.1f}%  {a['commit']:>7.1f}s"
                  f"   ({elapsed:.0f}s ~{eta:.0f}s left)")
            rows.append(dict(scenario=scn, variant=vlabel, **a))
        print()

    # Write CSV
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario","variant","sep",
                                          "side_pct","coll_pct","commit"])
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"Done. CSV -> {CSV_OUT}  (wall-time {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
