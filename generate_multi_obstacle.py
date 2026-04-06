#!/usr/bin/env python3
"""
generate_multi_obstacle.py
==========================
Run the three multi-vessel scenarios with CC-CBF, save:
  1. Static PNG plots  → paper/figures/multi_obstacle_<name>.png
  2. Animated GIFs     → gifs_paper/multi_<name>.gif

Scenarios
---------
  head_on_then_overtaking  : sequential Rule-14 then Rule-13
  parallel_head_on         : simultaneous Rule-14 × 2
  mixed_rules              : simultaneous Rule-14 + Rule-13

Usage
-----
    python generate_multi_obstacle.py
    python generate_multi_obstacle.py --scenario parallel_head_on
    python generate_multi_obstacle.py --no-gif
"""

import os, sys, io, time, math, argparse
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario import get_scenario
from core.simulator import Simulator
from core.entities import USVParameters
from core.colregs import classify_encounter
from core.geometry import center_distance
from algorithms.cc_cbf import (
    CCCBFController, R_BASE, LAMBDA, THETA_C, ACTIVATION_RANGE, _obs_rb,
)
import data.config as cfg

# ─────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR    = os.path.join(_HERE, "paper", "figures")
GIF_DIR     = os.path.join(_HERE, "gifs_paper")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(GIF_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Scenario registry
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "parallel_head_on": {
        "title":   "Simultaneous: Parallel Head-On × 2 (Rule 14 × 2)",
        "rules":   "Rule 14 × 2",
        "accent":  "#6A1B9A",
        "xlim":    (-110, 110),
        "ylim":    (-40, 40),
    },
    "mixed_rules": {
        "title":   "Simultaneous: Overtaking + Head-On (Rules 13 + 14)",
        "rules":   "Rule 13 + Rule 14",
        "accent":  "#1B5E20",
        "xlim":    (-110, 110),
        "ylim":    (-40, 40),
    },
}

SEED = 42
USV_COLOR    = "#1565C0"
TARGET_COLORS = ["#E53935", "#E65100", "#00838F"]   # one per target
GOAL_COLOR   = "#00C853"
BG_COLOR     = "#F8FAFC"

ENC_COLORS = {
    "head_on":            "#E53935",
    "overtaking":         "#1976D2",
    "crossing_give_way":  "#F57C00",
    "crossing_stand_on":  "#388E3C",
    "none":               "#BDBDBD",
}
ENC_LABELS = {
    "head_on":            "Head-On (R.14)",
    "overtaking":         "Overtaking (R.13)",
    "crossing_give_way":  "Crossing GW (R.15)",
    "crossing_stand_on":  "Crossing SO (R.17)",
    "none":               "No encounter",
}

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 120,
})

# GIF parameters
FPS        = 15
FRAME_SKIP = 3       # every 3rd sim step → 0.05 × 3 = 0.15 s/frame
MAX_FRAMES = 14 * FPS

# ─────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────
def run_sim(scenario_name: str, seed: int = SEED):
    sc = get_scenario(scenario_name, seed=seed)
    ctrl = CCCBFController()
    sim  = Simulator(sc, ctrl, USVParameters())
    sim.run_to_completion()
    return sim.records, sc


# ─────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────
def draw_ship(ax, cx, cy, psi, length, width, color,
              alpha=0.92, fill=True, lw=1.8, zorder=5):
    """Draw a simple ship hull polygon."""
    hl, hw = length / 2.0, width / 2.0
    bow    = hl * 0.6
    body   = np.array([
        [hl, 0], [bow, -hw], [-hl, -hw], [-hl, hw], [bow, hw], [hl, 0],
    ])
    c, s = np.cos(psi), np.sin(psi)
    R    = np.array([[c, -s], [s, c]])
    pts  = (R @ body.T).T + np.array([cx, cy])
    fc   = color if fill else "none"
    poly = plt.Polygon(pts, closed=True, ec=color, fc=fc,
                       alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(poly)


def draw_heading_arrow(ax, cx, cy, psi, length, color, lw=1.5, zorder=6):
    al   = length * 0.8
    dx   = al * math.cos(psi)
    dy   = al * math.sin(psi)
    ax.annotate("", xy=(cx+dx, cy+dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=zorder)


def gradient_trail(ax, xs, ys, color, lw=2.2, alpha_range=(0.08, 0.7),
                   zorder=3):
    """Draw a gradient-alpha polyline trail."""
    n = len(xs)
    if n < 2:
        return
    segs = [((xs[i], ys[i]), (xs[i+1], ys[i+1])) for i in range(n-1)]
    alphas = np.linspace(alpha_range[0], alpha_range[1], len(segs))
    for seg, a in zip(segs, alphas):
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                color=color, lw=lw, alpha=a, solid_capstyle='round', zorder=zorder)


def draw_cc_cbf_barrier_frame(ax, own_x, own_y, own_psi, enc_type, color,
                               alpha_fill=0.10, alpha_line=0.50):
    """Draw the CC-CBF directional barrier contour at the current position."""
    lam = LAMBDA.get(enc_type, 0.0)
    tc  = THETA_C.get(enc_type, 0.0)
    n   = 180
    angles = np.linspace(0, 2*np.pi, n)
    radii  = np.array([
        R_BASE * (1.0 + lam * max(0.0, math.cos(a - own_psi - tc)))
        for a in angles
    ])
    xs = own_x + radii * np.cos(angles)
    ys = own_y + radii * np.sin(angles)
    ax.fill(xs, ys, color=color, alpha=alpha_fill, zorder=2)
    ax.plot(xs, ys, color=color, lw=1.3, alpha=alpha_line, zorder=2)


# ─────────────────────────────────────────────────────────────────────
# Per-obstacle encounter breakdown helper
# ─────────────────────────────────────────────────────────────────────
def per_obstacle_encounter_counts(records):
    """Return {label: Counter(enc_type → count)} using r.encounters."""
    by_label = {}
    for r in records:
        for enc in r.encounters:
            lbl = enc.target_label
            by_label.setdefault(lbl, Counter())[enc.encounter_type] += 1
    return by_label


# ─────────────────────────────────────────────────────────────────────
# Static plot  — trajectory map only
# ─────────────────────────────────────────────────────────────────────
def make_static_plot(records, scenario, scenario_name, cfg_info):
    """Single-panel trajectory map (no right panels, no text box)."""
    n_targets = len(scenario.targets)
    tcolors   = TARGET_COLORS[:n_targets]

    # ── collect data ──
    ts  = [r.time for r in records]
    ds  = [r.min_distance for r in records]
    obs_labels = [t.label for t in scenario.targets]

    # CPA index
    cpa_idx = int(np.argmin(ds))

    # Figure — wide single panel
    fig, ax_map = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    # ══════════════════════════════════════════════════════════════════
    # Trajectory map
    # ══════════════════════════════════════════════════════════════════
    ax_map.set_facecolor("white")
    ax_map.grid(True, alpha=0.15, lw=0.4, color="#CCCCCC")

    # Obstacle trails
    for oi, (tgt, tc_) in enumerate(zip(scenario.targets, tcolors)):
        ox = [r.obstacles[oi].x for r in records]
        oy = [r.obstacles[oi].y for r in records]
        gradient_trail(ax_map, ox, oy, tc_, lw=1.6,
                       alpha_range=(0.08, 0.55), zorder=2)

    # USV trail — colour-coded by encounter type
    for i in range(len(records) - 1):
        r   = records[i]
        enc = r.encounter.encounter_type if r.encounter else "none"
        c   = ENC_COLORS.get(enc, "#BDBDBD")
        ax_map.plot([records[i].state.x, records[i+1].state.x],
                    [records[i].state.y, records[i+1].state.y],
                    color=c, lw=2.4, alpha=0.80, solid_capstyle='round', zorder=3)

    # Ship hulls at start, CPA, and end
    def _draw_snap(idx, alpha_own=0.9, alpha_obs=0.80, lw=1.6):
        r   = records[idx]
        own = r.state
        draw_ship(ax_map, own.x, own.y, own.psi,
                  cfg.USV_LENGTH, cfg.USV_WIDTH,
                  USV_COLOR, alpha=alpha_own, fill=True, lw=lw, zorder=6)
        draw_heading_arrow(ax_map, own.x, own.y, own.psi,
                           cfg.USV_LENGTH, USV_COLOR, lw=lw)
        for oi, tc_ in enumerate(tcolors):
            obs = r.obstacles[oi]
            draw_ship(ax_map, obs.x, obs.y, obs.psi,
                      obs.length, obs.width,
                      tc_, alpha=alpha_obs, fill=True, lw=lw, zorder=6)
            draw_heading_arrow(ax_map, obs.x, obs.y, obs.psi,
                               obs.length, tc_, lw=lw)

    _draw_snap(0,       alpha_own=0.35, alpha_obs=0.30, lw=1.0)   # start (faint)
    _draw_snap(cpa_idx, alpha_own=0.70, alpha_obs=0.65, lw=1.4)   # CPA
    _draw_snap(-1,      alpha_own=0.95, alpha_obs=0.85, lw=1.8)   # end (bright)

    # USV start label
    ax_map.annotate("USV start",
                    (records[0].state.x, records[0].state.y),
                    textcoords="offset points", xytext=(-8, 10),
                    fontsize=8, color=USV_COLOR, fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Obstacle start labels
    for oi, (tgt, tc_) in enumerate(zip(scenario.targets, tcolors)):
        ox0, oy0 = records[0].obstacles[oi].x, records[0].obstacles[oi].y
        ax_map.annotate(tgt.label.replace("_", " "),
                        (ox0, oy0), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=tc_,
                        fontweight='bold', zorder=10,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Legend
    legend_enc = [
        Line2D([0], [0], color=c, lw=3, label=ENC_LABELS[k])
        for k, c in ENC_COLORS.items()
        if any(
            (r.encounter.encounter_type == k if r.encounter else k == "none")
            for r in records
        )
    ]
    ax_map.legend(handles=legend_enc, loc='upper left', fontsize=13,
                  framealpha=0.92, ncol=1)

    ax_map.set_xlabel("x (m)", fontsize=17)
    ax_map.set_ylabel("y (m)", fontsize=17)
    ax_map.set_xlim(*cfg_info.get("xlim", (-200, 200)))
    ax_map.set_ylim(*cfg_info.get("ylim", (-40, 40)))
    ax_map.set_aspect("equal")

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"multi_obstacle_{scenario_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    out_eps = os.path.join(PLOT_DIR, f"multi_obstacle_{scenario_name}.eps")
    fig.savefig(out_eps, format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Plot saved: {out_path}")
    print(f"  ✓ EPS  saved: {out_eps}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# Separation distance — standalone plot
# ─────────────────────────────────────────────────────────────────────
def make_distance_plot(records, scenario, scenario_name, cfg_info):
    """Standalone separation distance vs time plot."""
    n_targets  = len(scenario.targets)
    tcolors    = TARGET_COLORS[:n_targets]
    obs_labels = [t.label for t in scenario.targets]

    ts  = [r.time for r in records]
    ds  = [r.min_distance for r in records]
    cpa_idx = int(np.argmin(ds))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG_COLOR)

    # Overall min-separation envelope
    ax.plot(ts, ds, color=USV_COLOR, lw=2.2, label="Min separation", zorder=4)

    # Per-obstacle distance curves
    for oi, (lbl, tc_) in enumerate(zip(obs_labels, tcolors)):
        d_per = [center_distance(r.state.x, r.state.y,
                                 r.obstacles[oi].x, r.obstacles[oi].y)
                 for r in records]
        ax.plot(ts, d_per, color=tc_, lw=1.4, ls='--', alpha=0.75,
                label=f"to {lbl.replace('_', ' ')}", zorder=3)

    # D_safe reference line
    d_safe_val = getattr(cfg, 'D_SAFE', 15.0)
    ax.axhline(d_safe_val, color='red', ls='--', lw=1.2, alpha=0.7,
               label=f"D$_{{safe}}$ = {d_safe_val:.0f} m")

    # CPA vertical marker
    ax.axvline(ts[cpa_idx], color='orange', ls=':', lw=1.3, alpha=0.85,
               label=f"CPA  t = {ts[cpa_idx]:.1f} s")
    ax.annotate(f"{ds[cpa_idx]:.1f} m",
                (ts[cpa_idx], ds[cpa_idx]),
                textcoords="offset points", xytext=(6, 5),
                fontsize=9, color='red', fontweight='bold')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Separation distance (m)")
    ax.set_title(f"Separation Distance  —  {cfg_info['title']}",
                 fontweight='bold', fontsize=11, pad=6)
    ax.legend(fontsize=8, framealpha=0.92, loc='upper right')
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, None)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"multi_obstacle_{scenario_name}_distance.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Distance plot saved: {out_path}")
    return out_path
    return out_path


# ─────────────────────────────────────────────────────────────────────
# GIF renderer – single frame
# ─────────────────────────────────────────────────────────────────────
def render_gif_frame(fig, ax, records, scenario, scenario_name,
                     frame_idx, cfg_info):
    ax.clear()

    n_targets = len(scenario.targets)
    tcolors   = TARGET_COLORS[:n_targets]
    obs_labels = [t.label for t in scenario.targets]

    rec = records[frame_idx]
    own = rec.state
    t   = rec.time
    ds_so_far = [records[i].min_distance for i in range(frame_idx + 1)]
    min_d = rec.min_distance

    # ── Axis limits (dynamic – follow action) ──
    all_x = [records[i].state.x for i in range(frame_idx + 1)]
    all_y = [records[i].state.y for i in range(frame_idx + 1)]
    for oi in range(n_targets):
        all_x += [records[i].obstacles[oi].x for i in range(frame_idx + 1)]
        all_y += [records[i].obstacles[oi].y for i in range(frame_idx + 1)]
    all_x.append(scenario.ownship_goal_x)
    all_y.append(scenario.ownship_goal_y)

    cx   = (min(all_x) + max(all_x)) / 2.0
    cy   = (min(all_y) + max(all_y)) / 2.0
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) + 60
    half = max(span, 100) / 2.0

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, alpha=0.20, lw=0.5, color="#AAAAAA")

    # ── Goal ──
    ax.plot(scenario.ownship_goal_x, scenario.ownship_goal_y,
            '*', color=GOAL_COLOR, ms=18, markeredgecolor='#00701A',
            markeredgewidth=1.2, zorder=9)
    ax.annotate("Goal",
                (scenario.ownship_goal_x, scenario.ownship_goal_y),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color='#00701A', fontweight='bold', zorder=9)

    # ── Obstacle trails ──
    for oi, tc_ in enumerate(tcolors):
        ox = [records[i].obstacles[oi].x for i in range(frame_idx + 1)]
        oy = [records[i].obstacles[oi].y for i in range(frame_idx + 1)]
        gradient_trail(ax, ox, oy, tc_, lw=1.5,
                       alpha_range=(0.08, 0.45), zorder=2)

    # ── USV trail (colour by encounter) ──
    for i in range(frame_idx):
        r   = records[i]
        enc = r.encounter.encounter_type if r.encounter else "none"
        c   = ENC_COLORS.get(enc, "#BDBDBD")
        ax.plot([records[i].state.x, records[i+1].state.x],
                [records[i].state.y, records[i+1].state.y],
                color=c, lw=2.2, alpha=0.80, solid_capstyle='round', zorder=3)

    # ── Safety circles ──
    for oi, tc_ in enumerate(tcolors):
        obs = rec.obstacles[oi]
        sr  = (math.sqrt(cfg.USV_LENGTH**2 + cfg.USV_WIDTH**2)
               + math.sqrt(obs.length**2 + obs.width**2)) / 2.0 + 2.0
        circ = Circle((obs.x, obs.y), sr,
                      fc=tc_, ec=tc_, alpha=0.08, lw=1.0, ls="--", zorder=1)
        ax.add_patch(circ)

    # ── Obstacle ships ──
    for oi, tc_ in enumerate(tcolors):
        obs = rec.obstacles[oi]
        draw_ship(ax, obs.x, obs.y, obs.psi,
                  obs.length, obs.width, tc_, alpha=0.88, zorder=6)
        draw_heading_arrow(ax, obs.x, obs.y, obs.psi,
                           obs.length, tc_, lw=1.3, zorder=7)
        ax.annotate(obs.label.replace("_", " "),
                    (obs.x, obs.y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5, color=tc_,
                    fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── CC-CBF barrier overlay ──
    enc_type = rec.encounter.encounter_type if rec.encounter else "none"
    draw_cc_cbf_barrier_frame(ax, own.x, own.y, own.psi, enc_type,
                               USV_COLOR, alpha_fill=0.10, alpha_line=0.45)

    # ── USV ship ──
    draw_ship(ax, own.x, own.y, own.psi,
              cfg.USV_LENGTH, cfg.USV_WIDTH,
              USV_COLOR, alpha=0.95, lw=2.2, zorder=7)
    draw_heading_arrow(ax, own.x, own.y, own.psi,
                       cfg.USV_LENGTH, USV_COLOR, lw=2.2, zorder=8)

    # ── Distance line to closest obstacle ──
    closest_oi = min(range(n_targets),
                     key=lambda i: center_distance(
                         own.x, own.y,
                         rec.obstacles[i].x, rec.obstacles[i].y))
    obs_c = rec.obstacles[closest_oi]
    ax.plot([own.x, obs_c.x], [own.y, obs_c.y],
            color='gray', ls=':', lw=0.9, alpha=0.55, zorder=2)
    mx, my = (own.x + obs_c.x) / 2, (own.y + obs_c.y) / 2
    d_color = "#F44336" if min_d < getattr(cfg, 'D_SAFE', 15.0) else "gray"
    ax.annotate(f"{min_d:.1f}m", (mx, my),
                textcoords="offset points", xytext=(4, 4),
                fontsize=8, color=d_color, fontweight='bold', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=d_color, alpha=0.85, lw=0.5))

    # ── CPA marker ──
    cpa_idx_so_far = int(np.argmin(ds_so_far))
    if frame_idx > cpa_idx_so_far + 5:
        cpax = records[cpa_idx_so_far].state.x
        cpay = records[cpa_idx_so_far].state.y
        ax.plot(cpax, cpay, 'o', color='red', ms=9, zorder=9)
        ax.plot(cpax, cpay, 'o', color='white', ms=4, zorder=10)
        ax.annotate(f"CPA {ds_so_far[cpa_idx_so_far]:.1f}m",
                    (cpax, cpay), textcoords="offset points",
                    xytext=(-12, -14), fontsize=7.5, color='red',
                    fontweight='bold', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='red', alpha=0.85, lw=0.5))

    # ── Per-obstacle encounter HUD (use r.encounters) ──
    enc_hud_lines = [f"t = {t:.1f} s"]
    lbl_to_enc = {e.target_label: e.encounter_type for e in rec.encounters}
    for lbl in obs_labels:
        et = lbl_to_enc.get(lbl, "none")
        nice = et.replace("_", " ").title()
        enc_hud_lines.append(f"  {lbl.replace('_',' ')}: {nice}")
    enc_hud_lines.append(f"Min sep: {min_d:.1f} m")
    enc_hud_lines.append(f"Speed: {own.u:.1f} m/s")

    hud = "\n".join(enc_hud_lines)
    ax.text(0.02, 0.98, hud, transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='#BDBDBD', alpha=0.93, lw=0.8),
            zorder=12)

    # Encounter colour dot next to HUD
    enc_dot_color = ENC_COLORS.get(enc_type, "#BDBDBD")
    ax.text(0.02, 0.64, "●", transform=ax.transAxes, fontsize=14,
            color=enc_dot_color, va='top', zorder=12)

    # ── Legend ──
    legend_els = [
        Line2D([0], [0], color=USV_COLOR, lw=2.5, label="USV (CC-CBF)"),
        Line2D([0], [0], marker='*', color=GOAL_COLOR, ms=10, lw=0, label='Goal'),
        Line2D([0], [0], color='red', ls='--', lw=1.0, label='CPA', marker='o', ms=5),
    ]
    for tc_, lbl in zip(tcolors, obs_labels):
        legend_els.append(Line2D([0], [0], color=tc_, lw=2.0,
                                 label=lbl.replace("_", " ")))
    for k, c in ENC_COLORS.items():
        if k != "none":
            legend_els.append(Line2D([0], [0], color=c, lw=2.5, alpha=0.7,
                                     label=ENC_LABELS[k]))
    ax.legend(handles=legend_els, loc='lower right', fontsize=6.5,
              framealpha=0.92, ncol=2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"CC-CBF  —  {cfg_info['title']}",
                 fontweight='bold', fontsize=12, pad=6)


# ─────────────────────────────────────────────────────────────────────
# GIF generator
# ─────────────────────────────────────────────────────────────────────
def make_gif(records, scenario, scenario_name, cfg_info):
    n = len(records)
    indices = list(range(0, n, FRAME_SKIP))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    if len(indices) > MAX_FRAMES:
        step = len(indices) / MAX_FRAMES
        indices = [indices[int(i * step)] for i in range(int(MAX_FRAMES))]
        if indices[-1] != n - 1:
            indices.append(n - 1)

    total = len(indices)
    print(f"  Rendering {total} frames @ {FPS} fps ...")

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout(pad=1.5)

    images = []
    for fi, fidx in enumerate(indices):
        render_gif_frame(fig, ax, records, scenario, scenario_name,
                         fidx, cfg_info)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()
        if (fi + 1) % 40 == 0 or fi == total - 1:
            print(f"    frame {fi+1}/{total}")

    plt.close(fig)

    # Freeze last frame for 1 s
    for _ in range(FPS):
        images.append(images[-1])

    out_path = os.path.join(GIF_DIR, f"multi_{scenario_name}.gif")
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    sz = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✓ GIF saved: {out_path}  ({sz:.1f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# Combined two-scenario subfigure
# ─────────────────────────────────────────────────────────────────────
def make_combined_plot(all_data):
    """Render both scenarios side-by-side as (a)/(b) subfigures.

    all_data : list of (records, scenario, scenario_name, cfg_info)
    """
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8))
    fig.patch.set_facecolor("white")

    subfig_labels = ["(a)", "(b)", "(c)"]

    for idx, (records, scenario, scenario_name, cfg_info) in enumerate(all_data):
        ax = axes[idx]
        n_targets = len(scenario.targets)
        tcolors   = TARGET_COLORS[:n_targets]

        ts = [r.time for r in records]
        ds = [r.min_distance for r in records]
        cpa_idx = int(np.argmin(ds))

        ax.set_facecolor(BG_COLOR)
        ax.grid(True, alpha=0.20, lw=0.5, color="#AAAAAA")

        # Obstacle trails
        for oi, (tgt, tc_) in enumerate(zip(scenario.targets, tcolors)):
            ox = [r.obstacles[oi].x for r in records]
            oy = [r.obstacles[oi].y for r in records]
            gradient_trail(ax, ox, oy, tc_, lw=1.6,
                           alpha_range=(0.08, 0.55), zorder=2)

        # USV trail — colour-coded by encounter type
        for i in range(len(records) - 1):
            r   = records[i]
            enc = r.encounter.encounter_type if r.encounter else "none"
            c   = ENC_COLORS.get(enc, "#BDBDBD")
            ax.plot([records[i].state.x, records[i+1].state.x],
                    [records[i].state.y, records[i+1].state.y],
                    color=c, lw=2.4, alpha=0.80,
                    solid_capstyle='round', zorder=3)

        # Safety circles
        for oi, tgt in enumerate(scenario.targets):
            obs_fin = records[-1].obstacles[oi]
            sr = (math.sqrt(cfg.USV_LENGTH**2 + cfg.USV_WIDTH**2)
                  + math.sqrt(obs_fin.length**2 + obs_fin.width**2)) / 2.0 + 2.0
            circ = Circle((obs_fin.x, obs_fin.y), sr,
                          fc=tcolors[oi], ec=tcolors[oi],
                          alpha=0.07, lw=1.0, ls="--", zorder=1)
            ax.add_patch(circ)

        # Ship hulls at start, CPA, end
        def _draw_snap(ax_, idx_, alpha_own=0.9, alpha_obs=0.80, lw=1.6):
            r_   = records[idx_]
            own_ = r_.state
            draw_ship(ax_, own_.x, own_.y, own_.psi,
                      cfg.USV_LENGTH, cfg.USV_WIDTH,
                      USV_COLOR, alpha=alpha_own, fill=True, lw=lw, zorder=6)
            draw_heading_arrow(ax_, own_.x, own_.y, own_.psi,
                               cfg.USV_LENGTH, USV_COLOR, lw=lw)
            for oi_, tc__ in enumerate(tcolors):
                obs_ = r_.obstacles[oi_]
                draw_ship(ax_, obs_.x, obs_.y, obs_.psi,
                          obs_.length, obs_.width,
                          tc__, alpha=alpha_obs, fill=True, lw=lw, zorder=6)
                draw_heading_arrow(ax_, obs_.x, obs_.y, obs_.psi,
                                   obs_.length, tc__, lw=lw)

        _draw_snap(ax, 0,       alpha_own=0.35, alpha_obs=0.30, lw=1.0)
        _draw_snap(ax, cpa_idx, alpha_own=0.70, alpha_obs=0.65, lw=1.4)
        _draw_snap(ax, -1,      alpha_own=0.95, alpha_obs=0.85, lw=1.8)

        # Obstacle start labels
        for oi, (tgt, tc_) in enumerate(zip(scenario.targets, tcolors)):
            ox0, oy0 = records[0].obstacles[oi].x, records[0].obstacles[oi].y
            ax.annotate(tgt.label.replace("_", " "),
                        (ox0, oy0), textcoords="offset points",
                        xytext=(6, 6), fontsize=13, color=tc_,
                        fontweight='bold', zorder=10,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground='white')])

        # Legend
        legend_enc = [
            Line2D([0], [0], color=c, lw=3, label=ENC_LABELS[k])
            for k, c in ENC_COLORS.items()
            if any(
                (r.encounter.encounter_type == k if r.encounter else k == "none")
                for r in records
            )
        ]
        ax.legend(handles=legend_enc, loc='upper left', fontsize=13,
                  framealpha=0.92, ncol=1)

        ax.set_xlabel("x (m)", fontsize=17)
        ax.set_ylabel("y (m)", fontsize=17)
        ax.set_xlim(*cfg_info.get("xlim", (-200, 200)))
        ax.set_ylim(*cfg_info.get("ylim", (-40, 40)))
        ax.set_aspect("equal")

        # Subfigure label bottom-left inside axes
        ax.text(0.02, 0.03, subfig_labels[idx], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left',
                color='black',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    fig.tight_layout(w_pad=3.0)

    out_png = os.path.join(PLOT_DIR, "multi_obstacle_combined.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    out_eps = os.path.join(PLOT_DIR, "multi_obstacle_combined.eps")
    fig.savefig(out_eps, format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Combined PNG saved: {out_png}")
    print(f"  ✓ Combined EPS saved: {out_eps}")
    return out_png


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-obstacle scenarios")
    parser.add_argument("--scenario", default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Run a single scenario (default: all 3)")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip GIF generation (plots only)")
    args = parser.parse_args()

    scen_keys = [args.scenario] if args.scenario else list(SCENARIOS.keys())

    print("=" * 65)
    print("  Multi-Obstacle CC-CBF  |  Plots + GIFs")
    print("=" * 65)

    t0_total = time.time()
    plot_paths = []
    gif_paths  = []

    for sname in scen_keys:
        cfg_info = SCENARIOS[sname]
        print(f"\n── {sname} ──")
        print(f"   {cfg_info['title']}")

        t0 = time.time()
        records, scenario = run_sim(sname)
        min_d     = min(r.min_distance for r in records)
        collision = any(r.collision for r in records)
        print(f"  Sim done: {len(records)} steps, {records[-1].time:.1f}s  "
              f"min_sep={min_d:.2f}m  collision={collision}")

        # Per-obstacle breakdown
        enc_by = per_obstacle_encounter_counts(records)
        for lbl, cnts in sorted(enc_by.items()):
            dominant = max(cnts, key=cnts.get)
            print(f"  [{lbl}]: dominant={dominant}  counts={dict(cnts)}")

        # Static plot
        p = make_static_plot(records, scenario, sname, cfg_info)
        plot_paths.append(p)

        # Separate distance plot
        pd = make_distance_plot(records, scenario, sname, cfg_info)
        plot_paths.append(pd)

        # GIF
        if not args.no_gif:
            g = make_gif(records, scenario, sname, cfg_info)
            gif_paths.append(g)

        print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n{'='*65}")
    print(f"  Finished in {time.time()-t0_total:.1f}s")
    print(f"  Plots → {PLOT_DIR}")
    print(f"  GIFs  → {GIF_DIR}")
    print(f"{'='*65}")
    for p in plot_paths:
        print(f"    📊 {os.path.basename(p)}")
    for g in gif_paths:
        sz = os.path.getsize(g) / (1024 * 1024)
        print(f"    🎞  {os.path.basename(g)}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
