#!/usr/bin/env python3
"""
Component ablation of CC-CBF (paper Table IV and Sec. VI-E).

Every variant runs on the same paired seeds (default 200 trials per
scenario, seeds 300-499, sigma_v = 1.0 m/s, sigma_psi = 10 deg) in the three
COLREG encounters.  Reported per variant: selected CPA-side rate (Side%,
with Wilson 95 % interval), mean minimum hull clearance (Sep) and collision
rate (Coll%), plus paired exact McNemar tests on the side outcome.

Variants (paper rows first):
  full                              Full CC-CBF
  no_direction                      theta_C = 0 (lambda retained)
  no_sternpass                      Rule-15 stern-passage factor off
  no_anticipation                   gamma = 0
  no_nominal_shaping                heading bias and stern goal off
  no_nominal_shaping_no_direction     ... and theta_C = 0
  no_nominal_shaping_mirrored         ... and theta_C -> -theta_C
  c3bf                              isotropic collision-cone baseline
  mirrored                          theta_C -> -theta_C (cited in the text)
Additional variants: ot_flip, stbd_lobes (and their no_nominal_shaping_*
forms), no_dir_no_stern, no_dir_no_anticipation, no_stern_no_anticipation.

Usage:
    python run_ablation.py                       # paper configuration
    python run_ablation.py --trials 40 --variants full no_direction c3bf
    python run_ablation.py --variants all

Output: results/ablation_trials.csv (per trial), results/ablation_summary.csv
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import algorithms.cc_cbf as ccbf_mod                          # noqa: E402
from algorithms.c3bf import C3BFController                    # noqa: E402
from algorithms.cc_cbf import CCCBFController                 # noqa: E402
from core.entities import USVParameters                       # noqa: E402
from core.evaluation import (                                 # noqa: E402
    add_common_args, correct_side, cpa_record, ensure_dir, mcnemar_exact,
    parallel_map, wilson_ci,
)
from core.scenario import COLREG_SCENARIOS, get_scenario      # noqa: E402
from core.simulator import Simulator                          # noqa: E402

SCENARIOS = list(COLREG_SCENARIOS)
N_TRIALS = 200
SEED_OFFSET = 300
SPEED_VAR = 1.0
HEADING_VAR = math.radians(10)
# Side%: a collision or a missing goal entry counts as not side-correct
STRICT_SIDE = True

# Barrier-parameter overrides per geometry variant (see algorithms.cc_cbf.overridden)
_THETA0 = dict(ccbf_mod.THETA_C)
GEOMETRY_VARIANTS = {
    "full":                     {},
    "no_direction":             {"theta_c": {k: 0.0 for k in _THETA0}},
    "no_sternpass":             {"stern_pass_lambda": 0.0},
    "no_anticipation":          {"gamma": 0.0},
    "mirrored":                 {"theta_c": {k: -v for k, v in _THETA0.items()}},
    # overtaking lobe on the starboard side only
    "ot_flip":                  {"theta_c": {"overtaking": -_THETA0["overtaking"]}},
    # crossing give-way and overtaking lobes flipped, head-on / stand-on unchanged
    "stbd_lobes":               {"theta_c": {k: -_THETA0[k] for k in ("crossing_give_way", "overtaking")}},
    "no_dir_no_stern":          {"theta_c": {k: 0.0 for k in _THETA0}, "stern_pass_lambda": 0.0},
    "no_dir_no_anticipation":   {"theta_c": {k: 0.0 for k in _THETA0}, "gamma": 0.0},
    "no_stern_no_anticipation": {"stern_pass_lambda": 0.0, "gamma": 0.0},
}
SHAPING_OFF = "no_nominal_shaping"

VARIANT_LABELS = {
    "full":                            "Full CC-CBF",
    "no_direction":                    "No direction",
    "no_sternpass":                    "No stern-pass",
    "no_anticipation":                 "No anticipation",
    "no_nominal_shaping":              "No nominal shaping",
    "no_nominal_shaping_no_direction": "  + no direction",
    "no_nominal_shaping_mirrored":     "  + mirrored lobe",
    "c3bf":                            "C3BF (isotropic)",
    "mirrored":                        "Mirrored lobe",
}
PAPER_VARIANTS = list(VARIANT_LABELS)
ALL_VARIANTS = (PAPER_VARIANTS
                + [v for v in GEOMETRY_VARIANTS if v not in PAPER_VARIANTS]
                + [f"{SHAPING_OFF}_{v}" for v in ("ot_flip", "stbd_lobes")])

# Paired McNemar comparisons reported in the paper text
COMPARISONS = [
    ("full", "no_direction"), ("full", "no_sternpass"), ("full", "no_anticipation"),
    ("full", "no_nominal_shaping"), ("full", "mirrored"), ("full", "c3bf"),
    ("no_nominal_shaping", "no_nominal_shaping_no_direction"),
    ("no_nominal_shaping", "no_nominal_shaping_mirrored"),
    ("no_nominal_shaping_no_direction", "no_nominal_shaping_mirrored"),
]


def parse_variant(variant):
    """Return (controller kind, barrier overrides, controller kwargs)."""
    if variant == "c3bf":
        return "c3bf", {}, {}
    kwargs = {}
    geometry = variant
    if variant == SHAPING_OFF or variant.startswith(SHAPING_OFF + "_"):
        # Layer-2 heading bias and Layer-3 stern reference goal disabled;
        # the barrier geometry and the tightening are retained.
        kwargs = dict(use_heading_bias=False, use_goal_shaping=False)
        geometry = variant[len(SHAPING_OFF) + 1:] or "full"
    if geometry not in GEOMETRY_VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant}")
    return "cc", GEOMETRY_VARIANTS[geometry], kwargs


def run_trial(scn, seed, variant):
    """Run one trial of *variant*; returns dict(sep, side, commit, coll)."""
    kind, overrides, kwargs = parse_variant(variant)
    with ccbf_mod.overridden(**overrides):
        sc = get_scenario(scn, seed=seed, speed_var=SPEED_VAR, heading_var=HEADING_VAR)
        ctrl = C3BFController() if kind == "c3bf" else CCCBFController(**kwargs)
        sim = Simulator(sc, ctrl, USVParameters())
        sim.run_to_completion()
    m = sim.get_metrics()
    side = correct_side(cpa_record(sim.records), scn) and (
        not STRICT_SIDE or (not m.collision_flag and sim.reached_goal))
    commit = m.manoeuvre_commit_time if m.manoeuvre_commit_time < 1e6 else 0.0
    return dict(sep=m.min_separation, side=int(side), commit=commit,
                coll=int(m.collision_flag))


# Backwards-compatible name
_run = run_trial


def _job(job):
    scn, seed, variant = job
    return dict(scenario=scn, variant=variant, seed=seed, **run_trial(scn, seed, variant))


def summarise(rows, scenarios, variants):
    """Print the ablation table and McNemar tests; return summary rows."""
    summary = []
    for scn in scenarios:
        print(f"\n{'=' * 84}\n  {scn}\n{'=' * 84}")
        print(f"  {'Variant':<34}{'Side%':>7}  {'95% CI':>13}  {'Sep (m)':>8}  "
              f"{'min Sep':>8}  {'Coll%':>6}")
        res = {}
        for v in variants:
            rs = [r for r in rows if r["scenario"] == scn and r["variant"] == v]
            if not rs:
                continue
            res[v] = rs
            n, k = len(rs), sum(r["side"] for r in rs)
            lo, hi = wilson_ci(k, n)
            s = dict(scenario=scn, variant=v, n=n, side_pct=round(100 * k / n, 1),
                     side_ci_lo=round(lo, 1), side_ci_hi=round(hi, 1),
                     sep=round(sum(r["sep"] for r in rs) / n, 2),
                     min_sep=round(min(r["sep"] for r in rs), 2),
                     coll_pct=round(100 * sum(r["coll"] for r in rs) / n, 1),
                     commit=round(sum(r["commit"] for r in rs) / n, 1))
            summary.append(s)
            print(f"  {VARIANT_LABELS.get(v, v):<34}{s['side_pct']:>6.1f}%  "
                  f"[{lo:5.1f},{hi:5.1f}]  {s['sep']:>8.2f}  {s['min_sep']:>8.2f}  "
                  f"{s['coll_pct']:>5.1f}%")
        pairs = [(a, b) for a, b in COMPARISONS if a in res and b in res]
        if pairs:
            print("  Paired exact McNemar tests on the side outcome (a-only / b-only):")
        for a, b in pairs:
            ra = {r["seed"]: r["side"] for r in res[a]}
            rb = {r["seed"]: r["side"] for r in res[b]}
            common = sorted(set(ra) & set(rb))
            n_ab = sum(1 for s in common if ra[s] and not rb[s])
            n_ba = sum(1 for s in common if rb[s] and not ra[s])
            print(f"    {a} vs {b}: {n_ab}/{n_ba}  p = {mcnemar_exact(n_ab, n_ba):.2g}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"paired trials per scenario and variant (default: {N_TRIALS})")
    ap.add_argument("--seed0", type=int, default=SEED_OFFSET,
                    help=f"first seed (default: {SEED_OFFSET})")
    ap.add_argument("--variants", nargs="+", default=PAPER_VARIANTS,
                    help="variants to run, or 'all' (default: the paper's Table IV rows)")
    ap.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    add_common_args(ap)
    args = ap.parse_args()

    variants = ALL_VARIANTS if args.variants == ["all"] else args.variants
    for v in variants:
        parse_variant(v)   # fail fast on typos
    jobs = [(s, args.seed0 + i, v) for s in args.scenarios for v in variants
            for i in range(args.trials)]
    print(f"CC-CBF ablation: {len(args.scenarios)} scenarios x {len(variants)} variants x "
          f"{args.trials} trials = {len(jobs)} runs  (speed_var={SPEED_VAR} m/s, "
          f"heading_var={math.degrees(HEADING_VAR):.0f} deg)")

    t0 = time.time()
    rows = parallel_map(_job, jobs, args.workers, chunksize=4)
    print(f"Simulations finished in {time.time() - t0:.0f} s")

    summary = summarise(rows, args.scenarios, variants)

    out = ensure_dir(args.out_dir)
    for name, data in (("ablation_trials.csv", rows), ("ablation_summary.csv", summary)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"\nCSV -> {os.path.join(out, 'ablation_trials.csv')}, "
          f"{os.path.join(out, 'ablation_summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
