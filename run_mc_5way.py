#!/usr/bin/env python3
"""
Monte Carlo evaluation – 5-way comparison (CC-CBF + 4 baselines incl. TC-CBF).

Runs N randomised trials × 4 scenarios × 5 controllers, writes CSV + summary.

Usage:
    python run_mc_5way.py                 # default 50 trials
    python run_mc_5way.py --trials 100
"""

import os, sys, csv, time, argparse, math, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario import get_scenario
from core.simulator import Simulator
from core.entities import USVParameters

from algorithms.cc_cbf import CCCBFController
from algorithms.c3bf import C3BFController
from algorithms.rule_based_colreg import RuleBasedCOLREGController
from algorithms.geometric_cri import GeometricCRIController
from algorithms.turning_circle_cbf import TurningCircleCBFController

import data.config as cfg

# ── Controller registry (paper order) ────────────────────────────────
CONTROLLERS = [
    ("CC-CBF",        CCCBFController),
    ("C3BF",          C3BFController),
    ("Rule-COLREG",   RuleBasedCOLREGController),
    ("Geo-CRI",       GeometricCRIController),
    ("TC-CBF",        TurningCircleCBFController),
]

SCENARIOS = ["head_on", "crossing_give_way", "overtaking", "static_obstacles"]

OFFSET_STD  = cfg.MONTE_CARLO_LATERAL_OFFSET_STD   # 5.0 m
SPEED_STD   = cfg.MONTE_CARLO_SPEED_STD            # 0.3 m/s
HEADING_STD = cfg.MONTE_CARLO_HEADING_STD           # ~5°

CSV_HEADER = [
    "controller", "scenario", "trial", "seed",
    "collision", "min_separation", "colreg_compliance",
    "path_efficiency", "avg_speed", "path_length",
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_single(ctrl_factory, scenario_name, seed, params):
    """Run one trial, return dict of metrics."""
    scenario = get_scenario(
        scenario_name,
        seed=seed,
        offset=OFFSET_STD,
        speed_var=SPEED_STD,
        heading_var=HEADING_STD,
    )
    ctrl = ctrl_factory()
    sim = Simulator(scenario, ctrl, params)
    sim.run_to_completion()
    m = sim.get_metrics()
    return {
        "collision":         int(m.collision_flag),
        "min_separation":    round(m.min_separation, 3),
        "colreg_compliance": round(m.colreg_compliance, 4),
        "path_efficiency":   round(m.path_efficiency, 4),
        "avg_speed":         round(m.avg_speed, 4),
        "path_length":       round(m.path_length, 2),
    }


def fmt(vals):
    if not vals:
        return "N/A"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return f"{m:.2f} ± {s:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    n = args.trials
    out_path = args.out or os.path.join(OUT_DIR, "results_mc_5way.csv")
    params = USVParameters()
    total = n * len(SCENARIOS) * len(CONTROLLERS)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  5-Way Monte Carlo: {n} trials                     ║")
    print(f"║  {len(SCENARIOS)} scenarios × {len(CONTROLLERS)} controllers = {total:4d} runs       ║")
    print(f"╚══════════════════════════════════════════════════╝")

    rows = []
    done = 0
    t0 = time.time()

    for scen in SCENARIOS:
        for ctrl_name, ctrl_cls in CONTROLLERS:
            for trial in range(n):
                seed = trial
                try:
                    result = run_single(ctrl_cls, scen, seed, params)
                except Exception as e:
                    print(f"  ERROR: {ctrl_name} × {scen} × seed={seed}: {e}")
                    result = {k: 0 for k in CSV_HEADER[4:]}
                    result["collision"] = 1

                row = {"controller": ctrl_name, "scenario": scen,
                       "trial": trial, "seed": seed}
                row.update(result)
                rows.append(row)
                done += 1

                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 1
                    eta = (total - done) / rate
                    print(f"  [{done:4d}/{total}] {ctrl_name:18s} × {scen:20s}"
                          f"  seed={seed:2d}  d_min={result['min_separation']:6.1f}"
                          f"  compl={result['colreg_compliance']:.2f}"
                          f"  ({elapsed:.0f}s, ~{eta:.0f}s left)")

    # ── Write CSV ───────────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n✓ CSV: {out_path}")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"{'Controller':18s} {'Scenario':20s} {'Coll%':>6s} "
          f"{'Min Sep (m)':>14s} {'Compliance':>14s} {'Efficiency':>14s}")
    print(f"{'='*120}")

    agg = {c: {"sep": [], "compl": [], "eff": [], "coll": []} for c, _ in CONTROLLERS}

    for scen in SCENARIOS:
        for ctrl_name, _ in CONTROLLERS:
            sub = [r for r in rows if r["controller"] == ctrl_name and r["scenario"] == scen]
            coll_pct = 100.0 * sum(r["collision"] for r in sub) / len(sub)
            sep   = [r["min_separation"] for r in sub]
            compl = [r["colreg_compliance"] for r in sub]
            eff   = [r["path_efficiency"] for r in sub]

            agg[ctrl_name]["sep"].extend(sep)
            agg[ctrl_name]["compl"].extend(compl)
            agg[ctrl_name]["eff"].extend(eff)
            agg[ctrl_name]["coll"].extend([r["collision"] for r in sub])

            print(f"{ctrl_name:18s} {scen:20s} {coll_pct:5.1f}% "
                  f"{fmt(sep):>14s} {fmt(compl):>14s} {fmt(eff):>14s}")
        print(f"{'-'*120}")

    # Grand averages
    print(f"\n{'Controller':18s} {'Coll%':>6s} {'Mean Sep':>10s} "
          f"{'Mean Compl':>12s} {'Mean Eff':>10s}")
    print(f"{'='*60}")
    for ctrl_name, _ in CONTROLLERS:
        a = agg[ctrl_name]
        cp = 100.0 * sum(a["coll"]) / len(a["coll"])
        print(f"{ctrl_name:18s} {cp:5.1f}%  {statistics.mean(a['sep']):8.2f}  "
              f"{statistics.mean(a['compl']):10.4f}  {statistics.mean(a['eff']):8.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
