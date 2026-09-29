#!/usr/bin/env python3
"""
Monte Carlo comparison of CC-CBF and the four baselines (paper Table VII).

N randomised trials x 4 scenarios x 5 controllers (default N = 50, i.e.
1000 runs).  Trial i uses scenario seed i with the perturbations of paper
Sec. VI-A (5 m head-on lateral offset, sigma_v = 0.3 m/s, sigma_psi = 5 deg).

Aggregation follows the paper: Eff. is averaged over goal-reaching trials;
the grand averages are means of the per-scenario means, and the grand COL
excludes the static-obstacle scenario (no give-way obligation).

Usage:
    python run_mc_5way.py                          # 50 trials, all cores
    python run_mc_5way.py --trials 10 --workers 4
    python run_mc_5way.py --controllers CC-CBF C3BF --scenarios head_on

Output: results/results_mc_5way.csv (one row per trial)
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.config as cfg                                     # noqa: E402
from algorithms.registry import CONTROLLERS as _REGISTRY      # noqa: E402
from core.entities import USVParameters                       # noqa: E402
from core.evaluation import (                                 # noqa: E402
    add_common_args, ensure_dir, fisher_exact, fmt_mean_sd, parallel_map,
)
from core.scenario import CANONICAL_SCENARIOS, get_scenario   # noqa: E402
from core.simulator import Simulator                          # noqa: E402

# (name, class) pairs in paper order
CONTROLLERS = list(_REGISTRY.items())
SCENARIOS = list(CANONICAL_SCENARIOS)

OFFSET_STD = cfg.MONTE_CARLO_LATERAL_OFFSET_STD    # 5.0 m
SPEED_STD = cfg.MONTE_CARLO_SPEED_STD              # 0.3 m/s
HEADING_STD = cfg.MONTE_CARLO_HEADING_STD          # 5 deg

CSV_HEADER = [
    "controller", "scenario", "trial", "seed",
    "collision", "min_separation", "colreg_compliance",
    "path_efficiency", "avg_speed", "path_length", "reached_goal", "runtime",
]


def run_single(ctrl_factory, scenario_name, seed, params=None):
    """Run one Monte Carlo trial and return its metrics as a dict."""
    scenario = get_scenario(
        scenario_name,
        seed=seed,
        offset=OFFSET_STD,
        speed_var=SPEED_STD,
        heading_var=HEADING_STD,
    )
    sim = Simulator(scenario, ctrl_factory(), params or USVParameters())
    sim.run_to_completion()
    m = sim.get_metrics()
    return {
        "collision":         int(m.collision_flag),
        "min_separation":    round(m.min_separation, 3),
        "colreg_compliance": round(m.colreg_compliance, 4),
        "path_efficiency":   round(m.path_efficiency, 4),
        "avg_speed":         round(m.avg_speed, 4),
        "path_length":       round(m.path_length, 2),
        "reached_goal":      int(sim.reached_goal),
    }


def _job(job):
    name, scenario_name, trial = job
    t = time.perf_counter()
    r = run_single(_REGISTRY[name], scenario_name, trial)
    r["runtime"] = round(time.perf_counter() - t, 3)
    return dict(controller=name, scenario=scenario_name, trial=trial, seed=trial, **r)


def summarise(rows, controllers, scenarios):
    """Print the per-scenario table and the grand averages of Table VII."""
    print(f"\n{'=' * 96}")
    print(f"{'Scenario':<18} {'Controller':<12} {'Coll.%':>6} {'Sep (m)':>14} "
          f"{'COL':>13} {'Eff.':>13} {'min Sep':>8}")
    print(f"{'=' * 96}")
    per = {}
    for scen in scenarios:
        for c in controllers:
            sub = [r for r in rows if r["controller"] == c and r["scenario"] == scen]
            if not sub:
                continue
            sep = [r["min_separation"] for r in sub]
            col = [r["colreg_compliance"] for r in sub]
            eff = [r["path_efficiency"] for r in sub if r["reached_goal"]]
            coll = [r["collision"] for r in sub]
            per[(c, scen)] = dict(sep=sep, col=col, eff=eff, coll=coll)
            print(f"{scen:<18} {c:<12} {100 * statistics.mean(coll):>5.1f}% "
                  f"{fmt_mean_sd(sep, 1):>14} {fmt_mean_sd(col):>13} {fmt_mean_sd(eff):>13} "
                  f"{min(sep):>8.2f}")
        print(f"{'-' * 96}")

    print("\nGrand averages (means of the per-scenario means; COL excludes static obstacles)")
    print(f"{'Controller':<12} {'Coll.%':>7} {'Sep (m)':>8} {'COL':>6} {'Eff.':>6}  Collisions")
    for c in controllers:
        cells = [per[(c, s)] for s in scenarios if (c, s) in per]
        coll = [x for p in cells for x in p["coll"]]
        col_means = [statistics.mean(per[(c, s)]["col"]) for s in scenarios
                     if (c, s) in per and s != "static_obstacles"]
        eff_means = [statistics.mean(p["eff"]) for p in cells if p["eff"]]
        print(f"{c:<12} {100 * statistics.mean(coll):>6.1f}% "
              f"{statistics.mean(statistics.mean(p['sep']) for p in cells):>8.1f} "
              f"{statistics.mean(col_means) if col_means else float('nan'):>6.2f} "
              f"{statistics.mean(eff_means) if eff_means else float('nan'):>6.2f}  "
              f"{sum(coll)}/{len(coll)}")

    # Collision counts of CC-CBF against each baseline (two-sided Fisher exact)
    ref = "CC-CBF"
    if ref in controllers:
        k0 = sum(r["collision"] for r in rows if r["controller"] == ref)
        n0 = sum(1 for r in rows if r["controller"] == ref)
        print(f"\nFisher exact test on collision counts ({ref}: {k0}/{n0})")
        for c in controllers:
            if c == ref:
                continue
            k = sum(r["collision"] for r in rows if r["controller"] == c)
            n = sum(1 for r in rows if r["controller"] == c)
            p = fisher_exact(k0, n0 - k0, k, n - k)
            print(f"  vs {c:<12} {k:>3}/{n:<4} p = {p:.2g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=50,
                    help="trials per controller and scenario (default: 50)")
    ap.add_argument("--controllers", nargs="+", choices=list(_REGISTRY), default=list(_REGISTRY))
    ap.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    ap.add_argument("--out", default=None,
                    help="CSV path (default: <out-dir>/results_mc_5way.csv)")
    add_common_args(ap)
    args = ap.parse_args()

    out_path = args.out or os.path.join(ensure_dir(args.out_dir), "results_mc_5way.csv")
    jobs = [(c, s, i) for s in args.scenarios for c in args.controllers
            for i in range(args.trials)]
    print(f"Monte Carlo: {args.trials} trials x {len(args.scenarios)} scenarios x "
          f"{len(args.controllers)} controllers = {len(jobs)} runs")

    t0 = time.time()
    rows = parallel_map(_job, jobs, args.workers)
    print(f"Simulations finished in {time.time() - t0:.0f} s")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {out_path}")

    summarise(rows, args.controllers, args.scenarios)
    return 0


if __name__ == "__main__":
    sys.exit(main())
