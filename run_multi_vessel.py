#!/usr/bin/env python3
"""
Monte Carlo statistics of the two multi-vessel scenarios (paper Sec. VI-G).

CC-CBF on the parallel head-on and mixed-rules scenarios, 50 randomised
trials each (sigma_v = 0.3 m/s, sigma_psi = 5 deg).  Per target, a trial is
side-correct when the target is on the own ship's port side at that
target's CPA (port-to-port for head-on; own ship on the target's starboard
side when overtaking); a trial is side-correct overall when every target
is, there is no collision and the goal disc is reached.

Usage:
    python run_multi_vessel.py
    python run_multi_vessel.py --trials 10

Output: results/multi_vessel_trials.csv
"""

import argparse
import csv
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.config as cfg                                # noqa: E402
from algorithms.cc_cbf import CCCBFController            # noqa: E402
from core.entities import USVParameters                  # noqa: E402
from core.evaluation import add_common_args, ensure_dir, parallel_map  # noqa: E402
from core.geometry import wrap_angle                     # noqa: E402
from core.scenario import get_scenario                   # noqa: E402
from core.simulator import Simulator                     # noqa: E402

SCENARIOS = ["parallel_head_on", "mixed_rules"]


def trial(job):
    """Run one trial; returns per-target and overall side outcomes."""
    scn, seed = job
    sc = get_scenario(scn, seed=seed, speed_var=cfg.MONTE_CARLO_SPEED_STD,
                      heading_var=cfg.MONTE_CARLO_HEADING_STD)
    sim = Simulator(sc, CCCBFController(), USVParameters())
    sim.run_to_completion()
    m = sim.get_metrics()
    per = {}
    for j, tgt in enumerate(sc.targets):
        best = None
        for rec in sim.records:
            if rec.state is None or len(rec.obstacles) <= j:
                continue
            o, t = rec.state, rec.obstacles[j]
            d = math.hypot(o.x - t.x, o.y - t.y)
            if best is None or d < best[0]:
                best = (d, wrap_angle(math.atan2(t.y - o.y, t.x - o.x) - o.psi) > 0)
        per[tgt.label] = best[1]
    reached = sim.reached_goal
    return dict(scenario=scn, seed=seed, coll=int(m.collision_flag), sep=m.min_separation,
                reached=int(reached), per=per,
                side=int(all(per.values()) and reached and not m.collision_flag))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=50, help="trials per scenario (default: 50)")
    add_common_args(ap)
    args = ap.parse_args()

    t0 = time.time()
    out = parallel_map(trial, [(s, i) for s in SCENARIOS for i in range(args.trials)],
                       args.workers)
    for s in SCENARIOS:
        rs = [r for r in out if r["scenario"] == s]
        seps = [r["sep"] for r in rs]
        n = len(rs)
        per_target = "  ".join(
            f"{lbl}: {100 * sum(r['per'][lbl] for r in rs) / n:.0f}% on own port"
            for lbl in rs[0]["per"])
        print(f"{s:<18} coll {100 * sum(r['coll'] for r in rs) / n:4.1f}%  "
              f"sep {statistics.mean(seps):.1f} ± {statistics.stdev(seps) if n > 1 else 0.0:.1f} "
              f"(min {min(seps):.1f})  reached {100 * sum(r['reached'] for r in rs) / n:.0f}%  "
              f"all-side {100 * sum(r['side'] for r in rs) / n:.0f}%  | {per_target}")
    print(f"[{time.time() - t0:.0f} s]")

    path = os.path.join(ensure_dir(args.out_dir), "multi_vessel_trials.csv")
    labels = sorted({lbl for r in out for lbl in r["per"]})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "collision", "min_separation", "reached_goal",
                     "all_side_correct"] + [f"port_at_cpa_{lbl}" for lbl in labels])
        for r in out:
            w.writerow([r["scenario"], r["seed"], r["coll"], round(r["sep"], 3), r["reached"],
                        r["side"]] + [int(r["per"][lbl]) if lbl in r["per"] else "" for lbl in labels])
    print(f"CSV -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
