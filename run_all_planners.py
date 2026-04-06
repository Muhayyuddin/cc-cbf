#!/usr/bin/env python3
"""
run_all_planners.py
===================
Quick deterministic evaluation of all 5 controllers across all 4 scenarios.

Runs a single fixed trial (seed=42) per controller × scenario — no Monte Carlo
randomisation. Useful for a fast sanity-check / smoke-test to confirm all
planners execute correctly and reproduce the paper's deterministic results.

Usage:
    python run_all_planners.py
    python run_all_planners.py --seed 0
    python run_all_planners.py --scenario head_on
    python run_all_planners.py --controller CC-CBF
"""

import os, sys, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario  import get_scenario
from core.simulator import Simulator
from core.entities  import USVParameters

from algorithms.cc_cbf            import CCCBFController
from algorithms.c3bf              import C3BFController
from algorithms.rule_based_colreg import RuleBasedCOLREGController
from algorithms.geometric_cri     import GeometricCRIController
from algorithms.turning_circle_cbf import TurningCircleCBFController

# ── Registry ──────────────────────────────────────────────────────────────────
ALL_CONTROLLERS = [
    ("CC-CBF",       CCCBFController),
    ("C3BF",         C3BFController),
    ("Rule-COLREG",  RuleBasedCOLREGController),
    ("Geo-CRI",      GeometricCRIController),
    ("TC-CBF",       TurningCircleCBFController),
]

ALL_SCENARIOS = [
    "head_on",
    "crossing_give_way",
    "overtaking",
    "static_obstacles",
]

SCENARIO_LABELS = {
    "head_on":           "Head-On          (Rule 14)",
    "crossing_give_way": "Crossing Give-Way (Rule 15)",
    "overtaking":        "Overtaking        (Rule 13)",
    "static_obstacles":  "Static Obstacles",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _status(coll, sep, dsafe=10.0):
    if coll:           return "❌ COLLISION"
    if sep < dsafe:    return "⚠  <D_safe"
    return             "✅ OK"


def run_one(ctrl_cls, scenario_name, seed, params):
    sc   = get_scenario(scenario_name, seed=seed)
    ctrl = ctrl_cls()
    sim  = Simulator(sc, ctrl, params)
    t0   = time.perf_counter()
    sim.run_to_completion()
    elapsed = time.perf_counter() - t0
    m = sim.get_metrics()
    return m, elapsed


def print_header(title):
    w = 100
    print()
    print("═" * w)
    print(f"  {title}")
    print("═" * w)
    print(f"  {'Controller':<16} {'Collision':<10} {'Min Sep (m)':<14} "
          f"{'COLREG%':<10} {'Efficiency':<12} {'Time (s)':<10} {'Status'}")
    print("─" * w)


def print_row(name, m, elapsed):
    status = _status(m.collision_flag, m.min_separation)
    print(f"  {name:<16} {'Yes' if m.collision_flag else 'No':<10} "
          f"{m.min_separation:<14.2f} {m.colreg_compliance*100:<10.1f} "
          f"{m.path_efficiency:<12.3f} {elapsed:<10.3f} {status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deterministic single-seed planner evaluation.")
    ap.add_argument("--seed",       type=int,  default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--scenario",   type=str,  default=None,
                    choices=ALL_SCENARIOS,
                    help="Run a single scenario only")
    ap.add_argument("--controller", type=str,  default=None,
                    choices=[n for n, _ in ALL_CONTROLLERS],
                    help="Run a single controller only")
    args = ap.parse_args()

    scenarios   = [args.scenario]   if args.scenario   else ALL_SCENARIOS
    controllers = [(n, c) for n, c in ALL_CONTROLLERS
                   if args.controller is None or n == args.controller]

    params = USVParameters()
    total  = len(scenarios) * len(controllers)
    errors = []

    print(f"\n  Seed: {args.seed}  |  "
          f"{len(scenarios)} scenario(s) × {len(controllers)} controller(s) = {total} run(s)")

    for scen in scenarios:
        print_header(SCENARIO_LABELS.get(scen, scen))
        for name, Cls in controllers:
            try:
                m, elapsed = run_one(Cls, scen, args.seed, params)
                print_row(name, m, elapsed)
            except Exception as e:
                print(f"  {name:<16} ERROR: {e}")
                errors.append((name, scen, str(e)))

    # ── Grand summary ─────────────────────────────────────────────────────────
    if len(scenarios) > 1 and len(controllers) > 1:
        print()
        print("═" * 100)
        print("  GRAND SUMMARY  (averaged across all scenarios, seed={})".format(args.seed))
        print("═" * 100)
        print(f"  {'Controller':<16} {'Coll%':<8} {'Avg Sep (m)':<14} "
              f"{'Avg COLREG%':<14} {'Avg Efficiency'}")
        print("─" * 100)

        # Re-run to collect aggregates (results are fast, no caching needed)
        for name, Cls in controllers:
            seps, compls, effs, colls = [], [], [], []
            for scen in scenarios:
                try:
                    m, _ = run_one(Cls, scen, args.seed, params)
                    seps.append(m.min_separation)
                    compls.append(m.colreg_compliance)
                    effs.append(m.path_efficiency)
                    colls.append(int(m.collision_flag))
                except Exception:
                    pass
            if seps:
                coll_pct = 100.0 * sum(colls) / len(colls)
                print(f"  {name:<16} {coll_pct:<8.1f} "
                      f"{sum(seps)/len(seps):<14.2f} "
                      f"{100*sum(compls)/len(compls):<14.1f} "
                      f"{sum(effs)/len(effs):.3f}")

    print()
    if errors:
        print(f"  ⚠  {len(errors)} error(s) occurred:")
        for name, scen, msg in errors:
            print(f"     {name} × {scen}: {msg}")
        sys.exit(1)
    else:
        print(f"  ✅  All {total} run(s) completed successfully.")
    print()


if __name__ == "__main__":
    main()
