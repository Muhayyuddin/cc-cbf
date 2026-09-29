#!/usr/bin/env python3
"""
Deterministic comparison of the five controllers on the four canonical
scenarios (paper Table III).

One fixed trial (seed 42, no Monte Carlo perturbation) per controller and
scenario.  Reports the minimum hull-to-hull clearance (Sep), the per-step
COLREG adherence (COL), the path ratio (Eff.) and the mean controller
computation per control update.  Doubles as a smoke test of the code base.

Usage:
    python run_all_planners.py
    python run_all_planners.py --scenario head_on --controller CC-CBF
    python run_all_planners.py --repeats 3        # median timing, as in the paper

Output: results/table3_seed<seed>.csv
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.config as cfg                                   # noqa: E402
from algorithms.registry import CONTROLLERS                 # noqa: E402
from core.entities import USVParameters                     # noqa: E402
from core.evaluation import add_common_args, ensure_dir, parallel_map  # noqa: E402
from core.scenario import CANONICAL_SCENARIOS, get_scenario  # noqa: E402
from core.simulator import Simulator                        # noqa: E402

SCENARIO_LABELS = {
    "head_on":           "Head-On           (Rule 14)",
    "crossing_give_way": "Crossing Give-Way (Rule 15)",
    "overtaking":        "Overtaking        (Rule 13)",
    "static_obstacles":  "Static Obstacles",
}
WIDTH = 104


def run_one(job):
    """Run one controller on one scenario; returns metrics and timings."""
    name, scenario_name, seed = job
    sim = Simulator(get_scenario(scenario_name, seed=seed), CONTROLLERS[name](),
                    USVParameters())
    ctrl = sim.controller
    compute = ctrl.compute_command
    update_times = []

    def timed(*args, **kwargs):
        t = time.perf_counter()
        cmd = compute(*args, **kwargs)
        update_times.append(time.perf_counter() - t)
        return cmd

    ctrl.compute_command = timed
    t0 = time.perf_counter()
    sim.run_to_completion()
    wall = time.perf_counter() - t0
    m = sim.get_metrics()
    return dict(
        controller=name, scenario=scenario_name, seed=seed,
        collision=int(m.collision_flag),
        min_separation=m.min_separation,
        colreg_compliance=m.colreg_compliance,
        path_efficiency=m.path_efficiency,
        reached_goal=int(sim.reached_goal),
        ctrl_mean_us=1e6 * statistics.mean(update_times),
        ctrl_max_ms=1e3 * max(update_times),
        run_time_s=wall,
    )


def status(r):
    if r["collision"]:
        return "COLLISION"
    if r["min_separation"] < cfg.D_SAFE:
        return f"< D_safe ({cfg.D_SAFE:.0f} m)"
    if not r["reached_goal"]:
        return "goal not reached"
    return "OK"


def print_header(title):
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)
    print(f"  {'Controller':<13} {'Coll.':<6} {'Sep (m)':>8} {'COL':>6} {'Eff.':>7} "
          f"{'Ctrl (us)':>10} {'Max (ms)':>9} {'Run (s)':>8}   Status")
    print("-" * WIDTH)


def print_row(r):
    eff = f"{r['path_efficiency']:.3f}" if r["reached_goal"] else "  --- "
    print(f"  {r['controller']:<13} {'yes' if r['collision'] else 'no':<6} "
          f"{r['min_separation']:>8.2f} {r['colreg_compliance']:>6.2f} {eff:>7} "
          f"{r['ctrl_mean_us']:>10.1f} {r['ctrl_max_ms']:>9.2f} {r['run_time_s']:>8.2f}   {status(r)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="scenario seed (default: 42)")
    ap.add_argument("--scenario", choices=CANONICAL_SCENARIOS, default=None,
                    help="run a single scenario only")
    ap.add_argument("--controller", choices=list(CONTROLLERS), default=None,
                    help="run a single controller only")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repetitions per run; timings are the median (default: 1)")
    add_common_args(ap)
    ap.set_defaults(workers=1)
    args = ap.parse_args()

    scenarios = [args.scenario] if args.scenario else CANONICAL_SCENARIOS
    controllers = [args.controller] if args.controller else list(CONTROLLERS)
    jobs = [(c, s, args.seed) for s in scenarios for c in controllers]
    print(f"\n  Seed {args.seed}: {len(scenarios)} scenario(s) x {len(controllers)} "
          f"controller(s) = {len(jobs)} run(s)"
          + (f", {args.repeats} repeats" if args.repeats > 1 else ""))
    if args.workers != 1:
        print("  Note: timings are inflated when runs execute in parallel.")

    runs = parallel_map(run_one, jobs * args.repeats, args.workers)
    results = {}
    for r in runs:
        results.setdefault((r["controller"], r["scenario"]), []).append(r)
    rows = []
    for key, reps in results.items():
        row = dict(reps[0])
        for k in ("ctrl_mean_us", "ctrl_max_ms", "run_time_s"):
            row[k] = statistics.median(x[k] for x in reps)
        rows.append(row)

    for scen in scenarios:
        print_header(SCENARIO_LABELS[scen])
        for r in rows:
            if r["scenario"] == scen:
                print_row(r)

    if len(scenarios) > 1:
        print()
        print("=" * WIDTH)
        print("  AVERAGE  (COL excludes static obstacles; Eff. excludes runs that "
              "did not reach the goal)")
        print("=" * WIDTH)
        print(f"  {'Controller':<13} {'Coll.%':>6} {'Sep (m)':>8} {'COL':>6} {'Eff.':>7} "
              f"{'Ctrl (us)':>10}")
        print("-" * WIDTH)
        for name in controllers:
            sub = [r for r in rows if r["controller"] == name]
            col = [r["colreg_compliance"] for r in sub if r["scenario"] != "static_obstacles"]
            eff = [r["path_efficiency"] for r in sub if r["reached_goal"]]
            print(f"  {name:<13} {100 * statistics.mean(r['collision'] for r in sub):>6.1f} "
                  f"{statistics.mean(r['min_separation'] for r in sub):>8.2f} "
                  f"{statistics.mean(col) if col else float('nan'):>6.2f} "
                  f"{statistics.mean(eff) if eff else float('nan'):>7.3f} "
                  f"{statistics.mean(r['ctrl_mean_us'] for r in sub):>10.1f}")

    out = os.path.join(ensure_dir(args.out_dir), f"table3_seed{args.seed}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
