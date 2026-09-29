#!/usr/bin/env python3
"""
Solver-branch statistics of CC-CBF (paper Sec. VI-H, Algorithm 1).

Counts, over the Monte Carlo trials (50 x 4 scenarios, Table VII setting)
and the ablation trials (200 x 3 encounters, Table IV setting), how often
each branch of the constraint-priority solver is taken:

  full    -- tightened QP feasible (full anticipatory tightening retained)
  scaled  -- tightening relaxed by a uniform factor (barrier condition kept)
  infeas  -- barrier condition infeasible; least-violating command

and how many control updates see a sampled barrier value h_CC < 0.

Usage:
    python run_solver_stats.py
    python run_solver_stats.py --mc-trials 10 --ablation-trials 20
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import algorithms.cc_cbf as ccbf_mod                      # noqa: E402
import run_ablation as ablation                           # noqa: E402
import run_mc_5way as mc                                  # noqa: E402
from core.entities import USVParameters                   # noqa: E402
from core.evaluation import add_common_args, parallel_map  # noqa: E402
from core.scenario import get_scenario                    # noqa: E402
from core.simulator import Simulator                      # noqa: E402

FIELDS = ("steps", "full", "scaled", "infeas", "hneg")


def run_counted(job):
    """Run one CC-CBF trial while counting solver branches."""
    kind, scn, i = job
    cnt = dict.fromkeys(FIELDS, 0)
    solve = ccbf_mod._solve_s

    def counting_solve(ux, uy, rows4, v_max):
        cnt["steps"] += 1
        if ccbf_mod._qp2d_s(ux, uy, [(r[0], r[1], r[2]) for r in rows4], v_max) is not None:
            cnt["full"] += 1
        elif ccbf_mod._qp2d_s(ux, uy, [(r[0], r[1], r[3]) for r in rows4], v_max) is not None:
            cnt["scaled"] += 1
        else:
            cnt["infeas"] += 1
        return solve(ux, uy, rows4, v_max)

    if kind == "mc":
        sc = get_scenario(scn, seed=i, offset=mc.OFFSET_STD, speed_var=mc.SPEED_STD,
                          heading_var=mc.HEADING_STD)
    else:
        sc = get_scenario(scn, seed=ablation.SEED_OFFSET + i, speed_var=ablation.SPEED_VAR,
                          heading_var=ablation.HEADING_VAR)
    ccbf_mod._solve_s = counting_solve
    try:
        ctrl = ccbf_mod.CCCBFController()
        sim = Simulator(sc, ctrl, USVParameters())
        while not sim.finished:
            sim.step()
            if ctrl._h_vals and min(ctrl._h_vals.values()) < 0:
                cnt["hneg"] += 1
    finally:
        ccbf_mod._solve_s = solve
    return kind, scn, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-trials", type=int, default=50)
    ap.add_argument("--ablation-trials", type=int, default=200)
    add_common_args(ap, out_dir=None)
    args = ap.parse_args()

    t0 = time.time()
    jobs = ([("mc", s, i) for s in mc.SCENARIOS for i in range(args.mc_trials)]
            + [("abl", s, i) for s in ablation.SCENARIOS for i in range(args.ablation_trials)])
    out = parallel_map(run_counted, jobs, args.workers, chunksize=4)

    tot = {}
    for kind, scn, c in out:
        d = tot.setdefault((kind, scn), dict(runs=0, **dict.fromkeys(FIELDS, 0),
                                             runs_scaled=0, runs_infeas=0, runs_hneg=0))
        d["runs"] += 1
        for f in FIELDS:
            d[f] += c[f]
        d["runs_scaled"] += c["scaled"] > 0
        d["runs_infeas"] += c["infeas"] > 0
        d["runs_hneg"] += c["hneg"] > 0
    for k, d in sorted(tot.items()):
        print(k, d)
    a = {f: sum(d[f] for d in tot.values()) for f in next(iter(tot.values()))}
    print("ALL", a)
    if a["steps"]:
        print(f"full tightening retained in {100 * a['full'] / a['steps']:.2f} % of "
              f"{a['steps']} solver calls; h_CC < 0 in {a['hneg']} updates "
              f"({100 * a['hneg'] / a['steps']:.2f} %)")
    print(f"[{time.time() - t0:.0f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
