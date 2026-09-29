#!/usr/bin/env python3
"""
Fig. 8: barrier value h_CC(t) along the four canonical CC-CBF trajectories,
with the encounter-classification switch instants marked.

For each canonical scenario one deterministic trial (seed 42, as Table III) is
run and, at every control step, we log

  * h_CC as actually enforced in the QP (CCCBFController._cbf_constraint,
    i.e. including the Rule-15 stern-pass inflation where active),
  * the directional barrier  d^2 - [R_b(1 + lam*Phi)]^2  of Eq. (21),
    the object for which the jump bound (22) is derived,
  * the encounter class tau of the barrier-critical (minimum-h) obstacle,
  * the separation d.

Switch instants t_k are steps at which tau of the barrier-critical
obstacle changes while that obstacle is detected.  At each t_k we check
the post-switch condition (23), h_CC(t_k^+) >= 0, and compare the jump
|h^+ - h^-| with the analytic bound (22), 2 R_b^2 (1+lam_bar) lam_bar.

Configuration
-------------
The default reproduces the manuscript figure: CC-CBF's switching signal
uses the 2 deg classification dead-band of Remark 1 (core.colregs).
  --classifier deadband | memoryless   (memoryless = the classifier that
                                        generated Tables III-VII)

Usage
-----
  python generate_hcc_traces.py
  python generate_hcc_traces.py --classifier memoryless

Outputs
-------
  figures/hcc_traces[<suffix>].png / .pdf
  results/results_hcc_traces[<suffix>].csv   (per-step log)
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402
from matplotlib.patches import Patch                           # noqa: E402

import algorithms.cc_cbf as ccbf                               # noqa: E402
import core.colregs as colregs                                 # noqa: E402
import data.config as cfg                                      # noqa: E402
from algorithms.cc_cbf import CCCBFController, _obs_rb, _phi_and_dphi  # noqa: E402
from core.colregs import classify_encounter                    # noqa: E402
from core.entities import EncounterInfo, USVParameters         # noqa: E402
from core.evaluation import FIGURES_DIR, RESULTS_DIR, ensure_dir  # noqa: E402
from core.scenario import get_scenario                         # noqa: E402
from core.simulator import Simulator                           # noqa: E402

LAMBDA_BAR = max(ccbf.LAMBDA.values())          # 0.60

PANELS = [
    ("head_on",           "Head-on (R.14)"),
    ("crossing_give_way", "Crossing give-way (R.15)"),
    ("overtaking",        "Overtaking (R.13)"),
    ("static_obstacles",  "Static obstacles"),
]

TAU_COLOR = {
    "none":              "#eef1f5",
    "head_on":           "#d4e4f5",
    "crossing_give_way": "#fbe0cc",
    "crossing_stand_on": "#efe0f3",
    "overtaking":        "#d9eedf",
}
TAU_SHORT = {
    "none": "none", "head_on": "head-on",
    "crossing_give_way": "GW", "crossing_stand_on": "SO",
    "overtaking": "overtak.",
}

LINTHRESH   = 50.0    # symlog: linear on [-50, 50] m^2, logarithmic beyond
CLUSTER_GAP = 1.0     # s: switches closer than this form one alternation cluster
CLUSTER_MIN = 4       # switches needed before a cluster is drawn as a band


def _h_nominal(state, obs, enc_type):
    """Eq. (21) barrier: d^2 - [R_b (1 + lam*Phi)]^2  (no stern-pass factor)."""
    dx, dy = obs.x - state.x, obs.y - state.y
    d2 = dx * dx + dy * dy
    rb = _obs_rb(obs.length, obs.width)
    theta = math.atan2(math.sin(math.atan2(dy, dx) - state.psi),
                       math.cos(math.atan2(dy, dx) - state.psi))
    phi, _ = _phi_and_dphi(theta, enc_type)
    rcc = rb * (1.0 + ccbf.LAMBDA.get(enc_type, 0.0) * phi)
    return d2 - rcc * rcc, math.sqrt(d2), rb


def run_trace(scn_key):
    """Run one deterministic trial, logging the barrier along the trajectory."""
    sc   = get_scenario(scn_key, seed=42)
    ctrl = CCCBFController()
    sim  = Simulator(sc, ctrl, USVParameters())

    log  = []
    orig = ctrl.compute_command

    def wrapped(state, obstacles, encounters, goal_x, goal_y, time, dt,
                scenario_config=None):
        # Snapshot the controller's switching signal tau *before* this step
        # so that re-classifying below reproduces exactly the class the
        # controller uses for every detected target.
        tau_before = dict(ctrl._tau)
        cmd = orig(state=state, obstacles=obstacles, encounters=encounters,
                   goal_x=goal_x, goal_y=goal_y, time=time, dt=dt,
                   scenario_config=scenario_config)
        # Evaluate the barrier against every obstacle (ground truth) so the
        # trace is continuous outside the LiDAR range; the barrier-critical
        # obstacle is the one with the smallest enforced h.
        best = None
        tau_used = dict(ctrl._tau)   # class the controller used this step
        for obs in sim.obstacles:
            if obs.label in tau_used:
                enc = EncounterInfo(encounter_type=tau_used[obs.label])
            else:
                enc = classify_encounter(state, obs,
                                         prev_type=tau_before.get(obs.label))
            _, _, h_impl, _ = CCCBFController._cbf_constraint(state, obs, enc)
            h_nom, d, rb = _h_nominal(state, obs, enc.encounter_type)
            cand = dict(h_impl=h_impl, h_nom=h_nom, d=d, rb=rb,
                        tau=enc.encounter_type, label=obs.label,
                        detected=d <= cfg.LIDAR_RANGE)
            if best is None or h_impl < best["h_impl"]:
                best = cand
        log.append(dict(t=time, **best))
        return cmd

    ctrl.compute_command = wrapped
    sim.run_to_completion()
    return log


def switch_instants(log):
    return [i for i in range(1, len(log))
            if log[i]["tau"] != log[i - 1]["tau"] and log[i]["detected"]]


def tau_segments(log):
    segs, i0 = [], 0
    for i in range(1, len(log) + 1):
        if i == len(log) or log[i]["tau"] != log[i0]["tau"]:
            segs.append((i0, i, log[i0]["tau"]))
            i0 = i
    return segs


def switch_clusters(t, sw):
    """Group switch indices whose spacing is below CLUSTER_GAP."""
    clusters, cur = [], []
    for k in sw:
        if cur and t[k] - t[cur[-1]] > CLUSTER_GAP:
            clusters.append(cur); cur = []
        cur.append(k)
    if cur:
        clusters.append(cur)
    return clusters


def configure(args):
    colregs.HYSTERESIS_DEG = 0.0 if args.classifier == "memoryless" else 2.0
    return "_memoryless" if args.classifier == "memoryless" else ""


def main():
    ap = argparse.ArgumentParser(description="Fig. 8: barrier traces h_CC(t).")
    ap.add_argument("--classifier", default="deadband",
                    choices=["deadband", "memoryless"],
                    help="CC-CBF switching signal: 2 deg dead-band (paper) or memoryless")
    ap.add_argument("--fig-dir", default=FIGURES_DIR, help="figure directory (default: <repo>/figures)")
    ap.add_argument("--out-dir", default=RESULTS_DIR, help="CSV directory (default: <repo>/results)")
    args = ap.parse_args()
    suffix = configure(args)
    fig_dir = ensure_dir(args.fig_dir)
    plt.rcParams.update({
        "font.size": 8.0, "font.family": "serif", "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "mathtext.fontset": "cm",
    })

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.15))
    axes = axes.ravel()
    csv_rows, summary = [], []

    for ax, (key, title) in zip(axes, PANELS):
        log = run_trace(key)
        t  = np.array([r["t"] for r in log])
        h  = np.array([r["h_impl"] for r in log])
        hn = np.array([r["h_nom"] for r in log])
        d  = np.array([r["d"] for r in log])
        sw = switch_instants(log)
        for r in log:
            csv_rows.append(dict(scenario=key, **r))

        clusters = [c for c in switch_clusters(t, sw) if len(c) >= CLUSTER_MIN]
        clustered = {k for c in clusters for k in c}
        span = t[-1] - t[0]

        # encounter-class background bands (segments inside an alternation
        # cluster are too short to read and are replaced by the hatched band)
        n_lab = 0
        for i0, i1, tau in tau_segments(log):
            t0, t1 = t[i0], t[min(i1, len(t) - 1)]
            if t1 - t0 < 0.02 * span and any(c[0] <= i0 <= c[-1] for c in clusters):
                continue
            ax.axvspan(t0, t1, color=TAU_COLOR.get(tau, "#eef1f5"), lw=0, zorder=0)
            if t1 - t0 > 0.12 * span:
                y_lab = 0.965 if n_lab % 2 == 0 else 0.905   # stagger neighbours
                n_lab += 1
                ax.text(0.5 * (t0 + t1), y_lab, TAU_SHORT.get(tau, tau),
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=5.6, color="black", zorder=9)

        # dense alternation clusters: one hatched band + count
        for c in clusters:
            t0, t1 = t[c[0]], t[c[-1]]
            classes = sorted({log[k]["tau"] for k in c}, key=lambda s: TAU_SHORT[s])
            ax.axvspan(t0, t1, facecolor="#f3d9c4", edgecolor="#a35a1e",
                       hatch="////", lw=0.4, alpha=0.75, zorder=1)
            ax.text(0.5 * (t0 + t1), 0.86,
                    "$\\rightleftarrows$".join(TAU_SHORT[s] for s in classes)
                    + "\n" + rf"$\times${len(c)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=5.6, color="black", zorder=9, linespacing=1.0)

        # unsafe half-plane
        ax.axhspan(-LINTHRESH, 0, facecolor="#d64550", alpha=0.16, lw=0, zorder=1)
        ax.axhline(0.0, color="#b02a37", lw=0.7, zorder=4)

        ax.plot(t, h, color="#12395e", lw=1.0, zorder=6)

        for k in sw:
            if k not in clustered:
                ax.axvline(t[k], color="#7a1fa2", lw=0.55, ls=(0, (2.2, 1.5)),
                           alpha=0.85, zorder=3)
        ax.plot(t[sw], h[sw], "o", ms=2.0, mfc="#7a1fa2", mec="white",
                mew=0.35, ls="none", zorder=8)

        imin = int(np.argmin(h))
        ax.plot(t[imin], h[imin], "v", ms=3.2, mfc="#e8833a", mec="#7a3c0d",
                mew=0.4, zorder=8)

        ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.45)
        ax.set_yticks([0, 1e2, 1e3, 1e4])
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-LINTHRESH, max(h.max(), 1e4) * 4.0)
        ax.set_title(title, fontsize=7.4, pad=2.5)
        ax.grid(True, which="major", color="white", lw=0.45, alpha=0.9, zorder=2)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


        rb    = log[0]["rb"]
        bound = 2.0 * rb ** 2 * (1.0 + LAMBDA_BAR) * LAMBDA_BAR
        summary.append(dict(
            scenario=key, n_switch=len(sw), min_h=h.min(), min_d=d.min(),
            post_switch_ok=all(h[k] >= 0.0 for k in sw),
            min_post_h=(min(h[k] for k in sw) if sw else float("nan")),
            min_dwell=(min(t[sw[i + 1]] - t[sw[i]] for i in range(len(sw) - 1))
                       if len(sw) > 1 else float("nan")),
            max_jump_impl=(max(abs(h[k] - h[k - 1]) for k in sw) if sw else 0.0),
            max_jump_nom=(max(abs(hn[k] - hn[k - 1]) for k in sw) if sw else 0.0),
            bound=bound, rb=rb,
            clusters=[(len(c), t[c[0]], t[c[-1]]) for c in clusters]))

    for i in (0, 2):
        axes[i].set_ylabel(r"$h_{\mathrm{CC}}$ [m$^2$]", fontsize=8.0, labelpad=1.5)
    for i in (2, 3):
        axes[i].set_xlabel(r"$t$ [s]", fontsize=8.0, labelpad=1.0)

    handles = [
        Line2D([], [], color="#12395e", lw=1.0,
               label=r"$h_{\mathrm{CC}}(t)$"),
        Line2D([], [], color="#7a1fa2", lw=0.55, ls=(0, (2.2, 1.5)), marker="o",
               ms=2.0, mfc="#7a1fa2", mec="white", mew=0.35,
               label=r"switch $t_k$"),
        Patch(fc="#f3d9c4", ec="#a35a1e", hatch="////", lw=0.4,
              label=r"GW$\rightleftarrows$SO"),
        Line2D([], [], color="none", marker="v", ms=3.2, mfc="#e8833a",
               mec="#7a3c0d", mew=0.4, label=r"$\min h_{\mathrm{CC}}$"),
        Patch(fc="#d64550", alpha=0.16, ec="#b02a37", lw=0.5,
              label=r"$h_{\mathrm{CC}}<0$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.005),
               handlelength=1.5, handletextpad=0.5, columnspacing=1.0)

    fig.tight_layout(rect=(0, 0.075, 1, 1), pad=0.3)
    fig.subplots_adjust(hspace=0.55, wspace=0.32)

    png = os.path.join(fig_dir, f"hcc_traces{suffix}.png")
    fig.savefig(png, dpi=500)
    fig.savefig(os.path.join(fig_dir, f"hcc_traces{suffix}.pdf"))
    plt.close(fig)

    csv_out = os.path.join(ensure_dir(args.out_dir), f"results_hcc_traces{suffix}.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)

    print(f"figure -> {png}")
    print(f"log    -> {csv_out}\n")
    print(f"{'scenario':<20}{'#sw':>4}{'min h':>9}{'min d':>8}{'post>=0':>9}"
          f"{'min h(tk+)':>12}{'min dwell':>11}{'max|dh|impl':>13}"
          f"{'max|dh|(21)':>13}{'bound(22)':>11}{'R_b':>7}")
    for s in summary:
        print(f"{s['scenario']:<20}{s['n_switch']:>4}{s['min_h']:>9.1f}"
              f"{s['min_d']:>8.2f}{str(s['post_switch_ok']):>9}"
              f"{s['min_post_h']:>12.1f}{s['min_dwell']:>11.2f}"
              f"{s['max_jump_impl']:>13.1f}{s['max_jump_nom']:>13.1f}"
              f"{s['bound']:>11.1f}{s['rb']:>7.2f}")
        for n, t0, t1 in s["clusters"]:
            print(f"    alternation cluster: {n} switches on [{t0:.2f}, {t1:.2f}] s")


if __name__ == "__main__":
    main()
