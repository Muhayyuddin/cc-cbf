#!/usr/bin/env python3
"""
λ-sensitivity study for the CC-CBF paper
=============================================================

What λ actually controls
------------------------
λ scales Φ in  R_CC = R_base * (1 + λ·Φ(θ,τ)).

TWO distinct effects are measured:

 TABLE A — R_CC growth (λ sweep, fixed moderate noise):
   Shows R_CC grows with λ by design (mathematical verification).
   All λ values maintain 90-100% COLREG side compliance.
   C3BF baseline: ~55% side compliance (no directional encoding).

 TABLE B — Noise robustness (λ fixed at nominal, noise sweep):
   Shows CC-CBF maintains side% at high noise where C3BF collapses.
   This is the key result: directional barrier vs isotropic barrier.
"""

import os, sys, csv, time, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario   import get_scenario
from core.simulator  import Simulator
from core.entities   import USVParameters
from core.geometry   import wrap_angle

from algorithms.cc_cbf import CCCBFController, _phi_and_dphi, _obs_rb
import algorithms.cc_cbf as ccbf_mod
from algorithms.c3bf import C3BFController

# ── Configuration ─────────────────────────────────────────────────────
SCENARIOS        = ["head_on", "crossing_give_way", "overtaking"]
LAMBDA_VALUES    = [0.0, 0.3, 0.6, 0.9]
NOISE_LEVELS_DEG = [5, 10, 15, 20, 25]

SPEED_VAR       = 1.0
HEADING_VAR_NOM = math.radians(10)

N_TRIALS    = 40
SEED_OFFSET = 300

CSV_A = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results_lambda_sweep.csv")
CSV_B = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results_noise_robustness.csv")

_ORIG_LAMBDA = dict(ccbf_mod.LAMBDA)


# ── Helpers ────────────────────────────────────────────────────────────

def _correct_side(rec, scn):
    if rec.state is None or not rec.obstacles:
        return True
    own, obs = rec.state, rec.obstacles[0]
    dx, dy = obs.x - own.x, obs.y - own.y
    if scn in ("head_on", "overtaking"):
        return wrap_angle(math.atan2(dy, dx) - own.psi) > 0.0
    else:
        return ((own.x - obs.x) * math.cos(obs.psi) +
                (own.y - obs.y) * math.sin(obs.psi)) < 0.0


def _peak_rcc(records, scn):
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
        rb  = _obs_rb(obs.length, obs.width)
        rcc = rb * (1.0 + ccbf_mod.LAMBDA.get(scn, 0.0) * phi)
        if math.isnan(peak) or rcc > peak:
            peak = rcc
    return peak


def _set_lam(scn, lam):
    for k in ccbf_mod.LAMBDA:
        ccbf_mod.LAMBDA[k] = lam if k == scn else _ORIG_LAMBDA[k]


def _restore():
    for k in ccbf_mod.LAMBDA:
        ccbf_mod.LAMBDA[k] = _ORIG_LAMBDA[k]


def _trial_ccbf(scn, lam, seed, sv, hv):
    _set_lam(scn, lam)
    try:
        sc  = get_scenario(scn, seed=seed, speed_var=sv, heading_var=hv)
        sim = Simulator(sc, CCCBFController(), USVParameters())
        sim.run_to_completion()
        m   = sim.get_metrics()
        cpa = min(sim.records, key=lambda r: r.min_distance)
        return dict(sep=m.min_separation,
                    side=int(_correct_side(cpa, scn)),
                    rcc=_peak_rcc(sim.records, scn),
                    commit=m.manoeuvre_commit_time if m.manoeuvre_commit_time < 1e6 else 0.0,
                    coll=int(m.collision_flag))
    finally:
        _restore()


def _trial_c3bf(scn, seed, sv, hv):
    sc  = get_scenario(scn, seed=seed, speed_var=sv, heading_var=hv)
    sim = Simulator(sc, C3BFController(), USVParameters())
    sim.run_to_completion()
    m   = sim.get_metrics()
    cpa = min(sim.records, key=lambda r: r.min_distance)
    return dict(sep=m.min_separation,
                side=int(_correct_side(cpa, scn)),
                rcc=float('nan'),
                commit=m.manoeuvre_commit_time if m.manoeuvre_commit_time < 1e6 else 0.0,
                coll=int(m.collision_flag))


def _agg(rs):
    n = len(rs)
    rv = [r["rcc"] for r in rs if not math.isnan(r["rcc"])]
    return dict(
        sep      = round(sum(r["sep"]  for r in rs) / n, 2),
        side_pct = round(100 * sum(r["side"] for r in rs) / n, 1),
        rcc      = round(sum(rv) / len(rv), 2) if rv else float('nan'),
        commit   = round(sum(r["commit"] for r in rs) / n, 2),
        coll_pct = round(100 * sum(r["coll"] for r in rs) / n, 1),
    )


# ── Table A: λ sweep ──────────────────────────────────────────────────

def run_table_a(t0):
    print("\n" + "="*72)
    print(f"TABLE A: lambda sweep  (heading_var=10deg  speed_var={SPEED_VAR}  N={N_TRIALS})")
    print("="*72)
    print(f"  {'Scenario':<22}{'Ctrl':<10}{'lam':>4}  {'R_CC(m)':>8}  {'Sep(m)':>7}  {'Side%':>6}")
    print("  " + "-"*60)

    rows = []
    total = len(SCENARIOS) * (len(LAMBDA_VALUES) + 1) * N_TRIALS
    done  = 0

    for scn in SCENARIOS:
        res = [_trial_c3bf(scn, SEED_OFFSET+i, SPEED_VAR, HEADING_VAR_NOM)
               for i in range(N_TRIALS)]
        done += N_TRIALS
        a = _agg(res)
        print(f"  {scn:<22}{'C3BF':<10}{'---':>4}  {'---':>8}  {a['sep']:>7.2f}  {a['side_pct']:>5.1f}%")
        rows.append(dict(table="A", scenario=scn, ctrl="C3BF", lam="---", **a))

        for lam in LAMBDA_VALUES:
            res = [_trial_ccbf(scn, lam, SEED_OFFSET+i, SPEED_VAR, HEADING_VAR_NOM)
                   for i in range(N_TRIALS)]
            done += N_TRIALS
            a = _agg(res)
            rcc_s = f"{a['rcc']:.2f}" if not math.isnan(a['rcc']) else "  ---"
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  {scn:<22}{'CC-CBF':<10}{lam:>4.1f}  {rcc_s:>8}  {a['sep']:>7.2f}  {a['side_pct']:>5.1f}%"
                  f"  ({elapsed:.0f}s ~{eta:.0f}s left)")
            rows.append(dict(table="A", scenario=scn, ctrl="CC-CBF", lam=lam, **a))
        print()

    return rows


# ── Table B: noise robustness ─────────────────────────────────────────

def run_table_b(t0):
    print("\n" + "="*72)
    print(f"TABLE B: noise robustness  (lambda=nominal  speed_var={SPEED_VAR}  N={N_TRIALS})")
    print("="*72)
    print(f"  {'Scenario':<22}{'Noise':>7}  {'CC-CBF%':>9}  {'C3BF%':>7}  {'Delta%':>7}")
    print("  " + "-"*55)

    rows = []
    total = len(SCENARIOS) * len(NOISE_LEVELS_DEG) * 2 * N_TRIALS
    done  = 0

    for scn in SCENARIOS:
        nom_lam = _ORIG_LAMBDA.get(scn, 0.5)
        for ndeg in NOISE_LEVELS_DEG:
            hv = math.radians(ndeg)
            rcc = [_trial_ccbf(scn, nom_lam, SEED_OFFSET+i, SPEED_VAR, hv)
                   for i in range(N_TRIALS)]
            done += N_TRIALS
            rc3 = [_trial_c3bf(scn, SEED_OFFSET+i, SPEED_VAR, hv)
                   for i in range(N_TRIALS)]
            done += N_TRIALS
            acc, ac3 = _agg(rcc), _agg(rc3)
            delta = acc['side_pct'] - ac3['side_pct']
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  {scn:<22}{ndeg:>5}deg  {acc['side_pct']:>8.1f}%  {ac3['side_pct']:>6.1f}%  {delta:>+6.1f}%"
                  f"  ({elapsed:.0f}s ~{eta:.0f}s left)")
            rows.append(dict(table="B", scenario=scn, noise_deg=ndeg,
                             ccbf_side=acc['side_pct'], c3bf_side=ac3['side_pct'],
                             delta=delta, ccbf_sep=acc['sep'], c3bf_sep=ac3['sep'],
                             ccbf_coll=acc['coll_pct'], c3bf_coll=ac3['coll_pct']))
        print()

    return rows


# ── Main ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"CC-CBF sensitivity study  (N={N_TRIALS}  speed_var={SPEED_VAR})")

    rows_a = run_table_a(t0)
    rows_b = run_table_b(t0)

    with open(CSV_A, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table","scenario","ctrl","lam",
                                          "rcc","sep","side_pct","commit","coll_pct"])
        w.writeheader()
        for r in rows_a:
            w.writerow({k: r.get(k,"") for k in w.fieldnames})

    with open(CSV_B, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table","scenario","noise_deg",
                                          "ccbf_side","c3bf_side","delta",
                                          "ccbf_sep","c3bf_sep","ccbf_coll","c3bf_coll"])
        w.writeheader()
        for r in rows_b:
            w.writerow({k: r.get(k,"") for k in w.fieldnames})

    print(f"\n  Table A -> {CSV_A}")
    print(f"  Table B -> {CSV_B}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
