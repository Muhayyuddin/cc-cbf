#!/usr/bin/env python3
"""
Sensitivity of CC-CBF to lambda and to heading noise (paper Tables V-VI).

Table A (Table V): lambda sweep in {0, 0.3, 0.6, 0.9} for the encounter's own
    lambda, sigma_v = 1.0 m/s, sigma_psi = 10 deg, against C3BF.  R_max is the
    peak radius R_b (1 + lambda) of the default 8 x 3 m target; R_CC is the
    mean peak radius met along the trajectories.
Table B (Table VI): initial-heading noise sigma_psi in {5, ..., 25} deg at the
    nominal lambda: CPA-side rate of CC-CBF vs C3BF.

Both use 40 trials per cell on seeds 300-339.

Usage:
    python run_sensitivity.py
    python run_sensitivity.py --tables A --trials 10

Output: results/results_lambda_sweep.csv, results/results_noise_robustness.csv
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import algorithms.cc_cbf as ccbf_mod                              # noqa: E402
from algorithms.c3bf import C3BFController                        # noqa: E402
from algorithms.cc_cbf import CCCBFController, _obs_rb, _phi_and_dphi  # noqa: E402
from core.entities import USVParameters                           # noqa: E402
from core.evaluation import (                                     # noqa: E402
    add_common_args, correct_side, cpa_record, ensure_dir, parallel_map,
)
from core.geometry import wrap_angle                              # noqa: E402
from core.scenario import COLREG_SCENARIOS, get_scenario          # noqa: E402
from core.simulator import Simulator                              # noqa: E402

SCENARIOS = list(COLREG_SCENARIOS)
LAMBDA_VALUES = [0.0, 0.3, 0.6, 0.9]
NOISE_LEVELS_DEG = [5, 10, 15, 20, 25]

SPEED_VAR = 1.0
HEADING_VAR_NOM = math.radians(10)

N_TRIALS = 40
SEED_OFFSET = 300

_NOMINAL_LAMBDA = dict(ccbf_mod.LAMBDA)


def _peak_rcc(records, scn):
    """Largest directional radius R_CC met before CPA (first target)."""
    if not records:
        return float('nan')
    cpa_idx = min(range(len(records)), key=lambda i: records[i].min_distance)
    peak = float('nan')
    for rec in records[:cpa_idx + 1]:
        if rec.state is None or not rec.obstacles:
            continue
        own, obs = rec.state, rec.obstacles[0]
        dx, dy = obs.x - own.x, obs.y - own.y
        theta = wrap_angle(math.atan2(dy, dx) - own.psi)
        phi, _ = _phi_and_dphi(theta, scn)
        rcc = _obs_rb(obs.length, obs.width) * (1.0 + ccbf_mod.LAMBDA.get(scn, 0.0) * phi)
        if math.isnan(peak) or rcc > peak:
            peak = rcc
    return peak


def _result(sim, scn, rcc):
    m = sim.get_metrics()
    return dict(sep=m.min_separation,
                side=int(correct_side(cpa_record(sim.records), scn)
                         and not m.collision_flag and sim.reached_goal),
                rcc=rcc,
                commit=m.manoeuvre_commit_time if m.manoeuvre_commit_time < 1e6 else 0.0,
                coll=int(m.collision_flag))


def _trial_ccbf(scn, lam, seed, sv, hv):
    """CC-CBF trial with lambda of encounter *scn* set to *lam*."""
    with ccbf_mod.overridden(**{"lambda": {scn: lam}}):
        sc = get_scenario(scn, seed=seed, speed_var=sv, heading_var=hv)
        sim = Simulator(sc, CCCBFController(), USVParameters())
        sim.run_to_completion()
        return _result(sim, scn, _peak_rcc(sim.records, scn))


def _trial_c3bf(scn, seed, sv, hv):
    sc = get_scenario(scn, seed=seed, speed_var=sv, heading_var=hv)
    sim = Simulator(sc, C3BFController(), USVParameters())
    sim.run_to_completion()
    return _result(sim, scn, float('nan'))


def _job(job):
    ctrl, scn, lam, seed, hv = job
    if ctrl == "C3BF":
        return _trial_c3bf(scn, seed, SPEED_VAR, hv)
    return _trial_ccbf(scn, lam, seed, SPEED_VAR, hv)


def _agg(rs):
    n = len(rs)
    rv = [r["rcc"] for r in rs if not math.isnan(r["rcc"])]
    return dict(
        sep=round(sum(r["sep"] for r in rs) / n, 2),
        side_pct=round(100 * sum(r["side"] for r in rs) / n, 1),
        rcc=round(sum(rv) / len(rv), 2) if rv else float('nan'),
        commit=round(sum(r["commit"] for r in rs) / n, 2),
        coll_pct=round(100 * sum(r["coll"] for r in rs) / n, 1),
    )


def _run_cells(cells, trials, workers):
    """cells: list of (ctrl, scn, lam, heading_var); returns aggregated dicts."""
    jobs = [(c, s, lam, SEED_OFFSET + i, hv) for c, s, lam, hv in cells for i in range(trials)]
    out = parallel_map(_job, jobs, workers, chunksize=4)
    return [_agg(out[k * trials:(k + 1) * trials]) for k in range(len(cells))]


def run_table_a(trials, workers):
    print(f"\nTABLE A: lambda sweep  (heading_var=10 deg, speed_var={SPEED_VAR} m/s, N={trials})")
    print(f"  {'Scenario':<20}{'Controller':<10}{'lambda':>7}{'R_max':>8}{'R_CC':>8}"
          f"{'Sep (m)':>9}{'Side%':>8}{'Coll%':>7}")
    cells = []
    for scn in SCENARIOS:
        cells.append(("C3BF", scn, None, HEADING_VAR_NOM))
        cells += [("CC-CBF", scn, lam, HEADING_VAR_NOM) for lam in LAMBDA_VALUES]
    rows = []
    for (ctrl, scn, lam, _), a in zip(cells, _run_cells(cells, trials, workers)):
        r_max = ccbf_mod.R_BASE * (1.0 + lam) if lam is not None else float('nan')
        lam_s = "---" if lam is None else f"{lam:.1f}"
        r_max_s = "---" if lam is None else f"{r_max:.2f}"
        rcc_s = "---" if math.isnan(a["rcc"]) else f"{a['rcc']:.2f}"
        print(f"  {scn:<20}{ctrl:<10}{lam_s:>7}{r_max_s:>8}{rcc_s:>8}"
              f"{a['sep']:>9.2f}{a['side_pct']:>7.1f}%{a['coll_pct']:>6.1f}%")
        rows.append(dict(table="A", scenario=scn, ctrl=ctrl,
                         lam="---" if lam is None else lam,
                         r_max="---" if lam is None else round(r_max, 2), **a))
    return rows


def run_table_b(trials, workers):
    print(f"\nTABLE B: noise robustness  (nominal lambda, speed_var={SPEED_VAR} m/s, N={trials})")
    print(f"  {'Scenario':<20}{'Noise':>7}{'CC-CBF%':>9}{'C3BF%':>8}{'Delta':>8}")
    cells = []
    for scn in SCENARIOS:
        for ndeg in NOISE_LEVELS_DEG:
            hv = math.radians(ndeg)
            cells += [("CC-CBF", scn, _NOMINAL_LAMBDA[scn], hv), ("C3BF", scn, None, hv)]
    aggs = _run_cells(cells, trials, workers)
    rows = []
    for k in range(0, len(cells), 2):
        scn, hv = cells[k][1], cells[k][3]
        acc, ac3 = aggs[k], aggs[k + 1]
        ndeg = int(round(math.degrees(hv)))
        delta = acc['side_pct'] - ac3['side_pct']
        print(f"  {scn:<20}{ndeg:>5}deg{acc['side_pct']:>8.1f}%{ac3['side_pct']:>7.1f}%{delta:>+7.1f}")
        rows.append(dict(table="B", scenario=scn, noise_deg=ndeg,
                         ccbf_side=acc['side_pct'], c3bf_side=ac3['side_pct'],
                         delta=delta, ccbf_sep=acc['sep'], c3bf_sep=ac3['sep'],
                         ccbf_coll=acc['coll_pct'], c3bf_coll=ac3['coll_pct']))
    return rows


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="+", choices=["A", "B"], default=["A", "B"])
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"trials per cell (default: {N_TRIALS})")
    add_common_args(ap)
    args = ap.parse_args()
    out = ensure_dir(args.out_dir)

    t0 = time.time()
    if "A" in args.tables:
        _write(os.path.join(out, "results_lambda_sweep.csv"), run_table_a(args.trials, args.workers))
    if "B" in args.tables:
        _write(os.path.join(out, "results_noise_robustness.csv"), run_table_b(args.trials, args.workers))
    print(f"  Total time: {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
