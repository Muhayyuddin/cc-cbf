#!/usr/bin/env python3
"""
Generate animated GIFs for the 5 paper controllers across all 4 scenarios.

Controllers: CC-CBF, C3BF, Rule-COLREG, Geo-CRI, TC-CBF
Scenarios:   head_on, crossing_give_way, overtaking, static_obstacles

Each GIF shows:
  - Ownship trajectory (gradient trail) with ship hull
  - Target vessel(s) with trail and hull
  - Safety radius circle around each obstacle
  - Distance line to closest obstacle
  - CPA marker
  - HUD: time, speed, min distance, encounter type, COLREG compliance (running)
  - Real-time metrics panel

Usage:
    python generate_paper_gifs.py
    python generate_paper_gifs.py --controller CC-CBF --scenario head_on
"""

import os
import sys
import math
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from PIL import Image
import io

# ── project imports ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario import get_scenario
from core.simulator import Simulator
from core.entities import USVParameters
from core.colregs import classify_encounter
from core.geometry import center_distance
from core.metrics import compute_metrics

from algorithms.cc_cbf import (
    CCCBFController, R_BASE as CC_R_BASE, LAMBDA as CC_LAMBDA,
    THETA_C as CC_THETA_C,
)
from algorithms.c3bf import C3BFController, R_SAFE as C3BF_R_SAFE
from algorithms.geometric_cri import GeometricCRIController
from algorithms.rule_based_colreg import RuleBasedCOLREGController
from algorithms.turning_circle_cbf import (
    TurningCircleCBFController, R_TURN as TC_R_TURN,
)

import data.config as cfg

# =====================================================================
# Configuration
# =====================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs_paper")

SCENARIOS = ["head_on", "crossing_give_way", "overtaking", "static_obstacles"]
SCENARIO_TITLES = {
    "head_on":           "Head-On (Rule 14)",
    "crossing_give_way": "Crossing Give-Way (Rule 15)",
    "overtaking":        "Overtaking (Rule 13)",
    "static_obstacles":  "Static Obstacle Field",
}

SEED = 42
FPS = 15
FRAME_SKIP = 4          # render every Nth sim step (0.05s × 4 = 0.2s/frame)
GIF_DURATION_CAP = 14   # max GIF duration (seconds)

# Controller registry:  key → (class, display_name, color, suffix)
CONTROLLERS = {
    "CC-CBF": (CCCBFController,            "CC-CBF (Ours)",    "#1565C0", "cc_cbf"),
    "C3BF":   (C3BFController,             "C3BF",             "#E53935", "c3bf"),
    "Rule-COLREG": (RuleBasedCOLREGController, "Rule-COLREG",  "#43A047", "rule_colreg"),
    "Geo-CRI": (GeometricCRIController,    "Geo-CRI",          "#FB8C00", "geo_cri"),
    "TC-CBF":  (TurningCircleCBFController, "TC-CBF (Lee)",    "#7B1FA2", "tc_cbf"),
}

# Visual style
TARGET_COLOR   = "#F44336"
TARGET_TRAIL   = "#FFCDD2"
STATIC_COLOR   = "#9E9E9E"
SAFETY_COLOR   = "#4CAF50"
GOAL_COLOR     = "#00E676"
BG_COLOR       = "#FAFAFA"
D_SAFE_COLOR   = "#EF5350"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.dpi": 100,
})


# =====================================================================
# Drawing helpers
# =====================================================================

def draw_ship(ax, cx, cy, psi, length, width, color, alpha=0.9,
              fill=True, lw=1.5, zorder=5):
    hl, hw = length / 2.0, width / 2.0
    bow = hl * 0.6
    body = np.array([
        [hl, 0], [bow, -hw], [-hl, -hw], [-hl, hw], [bow, hw], [hl, 0],
    ])
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    pts = (R @ body.T).T + [cx, cy]
    fc = color if fill else "none"
    poly = plt.Polygon(pts, closed=True, ec=color, fc=fc,
                       alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(poly)


def draw_heading_arrow(ax, cx, cy, psi, length, color, lw=1.5, zorder=6):
    al = length * 0.7
    dx, dy = al * np.cos(psi), al * np.sin(psi)
    ax.annotate("", xy=(cx+dx, cy+dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=zorder)


def draw_safety_circle(ax, ox, oy, radius):
    c = Circle((ox, oy), radius, fc=SAFETY_COLOR, ec=SAFETY_COLOR,
               alpha=0.12, lw=1.0, ls="--", zorder=1)
    ax.add_patch(c)


def draw_dsafe_circle(ax, ox, oy):
    """Draw the D_safe boundary (dashed red) around obstacle."""
    c = Circle((ox, oy), cfg.D_SAFE, fc="none", ec=D_SAFE_COLOR,
               alpha=0.4, lw=1.0, ls=":", zorder=1)
    ax.add_patch(c)


# ── COLREG sector wedges (used by CC-CBF and Rule-COLREG) ────────────

_SECTOR_DEFS = [
    ("head_on",            -15,    15,    "#FF9800"),
    ("crossing_give_way",  -112.5, -15,   "#EF5350"),
    ("crossing_stand_on",   15,    112.5, "#66BB6A"),
    ("overtaking",          112.5, 247.5, "#42A5F5"),
]

def draw_colreg_sectors(ax, own_x, own_y, own_psi, radius,
                        encounter_type):
    """COLREG encounter sector wedges around ownship."""
    heading_deg = math.degrees(own_psi)
    for enc_name, s_off, e_off, base_clr in _SECTOR_DEFS:
        active = (encounter_type == enc_name)
        alpha = 0.22 if active else 0.06
        lw = 2.0 if active else 0.6
        theta1 = heading_deg + s_off
        theta2 = heading_deg + e_off
        w = Wedge((own_x, own_y), radius, theta1, theta2,
                  facecolor=base_clr, edgecolor=base_clr,
                  alpha=alpha, lw=lw, zorder=1)
        ax.add_patch(w)
        if active:
            mid_a = math.radians((theta1 + theta2) / 2.0)
            lbl_r = radius * 0.55
            lx = own_x + lbl_r * math.cos(mid_a)
            ly = own_y + lbl_r * math.sin(mid_a)
            nice = enc_name.replace("_", "\n").title()
            ax.text(lx, ly, nice, fontsize=6, ha='center', va='center',
                    color=base_clr, fontweight='bold', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2,
                                               foreground='white')])


def draw_colreg_rule_badge(ax, encounter_type, manoeuvre):
    """COLREG rule badge (top-right corner)."""
    rule_map = {
        "head_on": "Rule 14", "crossing_give_way": "Rule 15",
        "crossing_stand_on": "Rule 17", "overtaking": "Rule 13",
    }
    rule = rule_map.get(encounter_type, "—")
    nice = (manoeuvre.replace("_", " ").title()
            if manoeuvre else "Hold Course")
    txt = f"{rule}\n{nice}"
    clr = "#E65100" if manoeuvre not in ("hold_course", None) else "#4CAF50"
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=8,
            ha='right', va='top', fontweight='bold', color=clr,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=clr,
                      alpha=0.9, lw=1.2),
            zorder=12)


# ── CC-CBF barrier contour (R_CC polar shape) ────────────────────────

def draw_cc_cbf_barrier(ax, own_x, own_y, own_psi, encounter_type,
                        ctrl_color):
    """
    Draw the directional R_CC barrier contour around USV.
    R_CC(theta) = R_base * [1 + lambda * max(0, cos(theta - theta_C))]
    theta is relative to the obstacle bearing from own heading,
    but for visualisation we sweep all 360° in the WORLD frame.
    """
    lam = CC_LAMBDA.get(encounter_type, 0.0)
    tc  = CC_THETA_C.get(encounter_type, 0.0)

    n = 200
    angles = np.linspace(0, 2 * np.pi, n)
    radii = np.empty(n)
    for i, a in enumerate(angles):
        # theta_rel is the angle in body frame
        theta_body = a - own_psi
        phi = max(0.0, math.cos(theta_body - tc))
        radii[i] = CC_R_BASE * (1.0 + lam * phi)

    # Convert polar → Cartesian (in world frame)
    xs = own_x + radii * np.cos(angles)
    ys = own_y + radii * np.sin(angles)

    # Filled region
    ax.fill(xs, ys, alpha=0.12, color=ctrl_color, zorder=2)
    ax.plot(xs, ys, color=ctrl_color, lw=1.5, alpha=0.55, zorder=2)

    # Also draw isotropic R_base as dashed ring
    circ = Circle((own_x, own_y), CC_R_BASE, fc='none',
                  ec=ctrl_color, alpha=0.3, lw=0.8, ls='--', zorder=2)
    ax.add_patch(circ)

    # Label R_CC near the max
    max_angle = own_psi + tc
    r_max = CC_R_BASE * (1.0 + lam)
    lx = own_x + (r_max + 2.0) * math.cos(max_angle)
    ly = own_y + (r_max + 2.0) * math.sin(max_angle)
    if lam > 0.01:
        ax.text(lx, ly, r"$R_{CC}$", fontsize=7, ha='center', va='center',
                color=ctrl_color, fontweight='bold', zorder=8,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])


# ── C3BF isotropic barrier circle ────────────────────────────────────

def draw_c3bf_barrier(ax, own_x, own_y, ctrl_color):
    """Draw the isotropic R_safe barrier circle for C3BF."""
    circ = Circle((own_x, own_y), C3BF_R_SAFE,
                  fc=ctrl_color, ec=ctrl_color,
                  alpha=0.08, lw=1.2, ls='--', zorder=2)
    ax.add_patch(circ)
    ax.text(own_x + C3BF_R_SAFE + 1.5, own_y,
            r"$R_{safe}$", fontsize=7, ha='left', va='center',
            color=ctrl_color, fontweight='bold', zorder=8,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])


# ── Geo-CRI risk ring + gauge ────────────────────────────────────────

_CRI_THRESHOLD = 0.4

def draw_cri_ring(ax, own_x, own_y, cri_value, avoidance_domain):
    """Heat ring proportional to CRI risk around ownship."""
    risk_radius = avoidance_domain * max(0.15, cri_value)
    if cri_value > _CRI_THRESHOLD:
        heat_color = '#FF5722'
    elif cri_value > 0.2:
        heat_color = '#FFA726'
    else:
        heat_color = '#66BB6A'
    heat_alpha = 0.08 + 0.15 * cri_value
    c_heat = Circle((own_x, own_y), risk_radius,
                    facecolor=heat_color, edgecolor='none',
                    alpha=heat_alpha, zorder=1)
    ax.add_patch(c_heat)
    # Threshold ring
    thr_r = _CRI_THRESHOLD * avoidance_domain
    c_thr = Circle((own_x, own_y), thr_r,
                   facecolor='none', edgecolor='#E65100',
                   alpha=0.50, lw=1.5, ls='-.', zorder=1)
    ax.add_patch(c_thr)


def draw_cri_gauge(ax, cri_value):
    """Vertical CRI gauge bar (right side)."""
    bar_x, bar_y0, bar_h, bar_w = 0.95, 0.30, 0.35, 0.025
    ax.add_patch(plt.Rectangle(
        (bar_x - bar_w / 2, bar_y0), bar_w, bar_h,
        transform=ax.transAxes, facecolor='#EEEEEE', edgecolor='#BDBDBD',
        lw=0.8, zorder=11, clip_on=False))
    fill_h = bar_h * min(1.0, cri_value)
    fill_color = '#FF5722' if cri_value > _CRI_THRESHOLD else '#FFA726'
    ax.add_patch(plt.Rectangle(
        (bar_x - bar_w / 2, bar_y0), bar_w, fill_h,
        transform=ax.transAxes, facecolor=fill_color, edgecolor='none',
        alpha=0.85, zorder=11, clip_on=False))
    thr_y = bar_y0 + bar_h * _CRI_THRESHOLD
    ax.plot([bar_x - bar_w, bar_x + bar_w], [thr_y, thr_y],
            transform=ax.transAxes, color='#E65100', lw=1.5, zorder=12,
            clip_on=False)
    ax.text(bar_x, bar_y0 - 0.03, f"CRI\n{cri_value:.2f}",
            transform=ax.transAxes, fontsize=7, ha='center', va='top',
            fontweight='bold', color=fill_color, zorder=12)


# ── TC-CBF turning circles ───────────────────────────────────────────

def draw_turning_circles(ax, own_x, own_y, own_psi, ctrl_color):
    """Draw port and starboard turning circles around USV."""
    # Starboard center
    sx = own_x + TC_R_TURN * math.sin(own_psi)
    sy = own_y - TC_R_TURN * math.cos(own_psi)
    # Port center
    px = own_x - TC_R_TURN * math.sin(own_psi)
    py = own_y + TC_R_TURN * math.cos(own_psi)

    for cx, cy, lbl, ls_ in [(sx, sy, "STBD", '-'), (px, py, "PORT", '--')]:
        circ = Circle((cx, cy), TC_R_TURN,
                      fc=ctrl_color, ec=ctrl_color,
                      alpha=0.06, lw=1.2, ls=ls_, zorder=2)
        ax.add_patch(circ)
        ax.plot(cx, cy, '+', color=ctrl_color, ms=6, mew=1.0,
                alpha=0.5, zorder=3)
        ax.text(cx, cy + TC_R_TURN + 1.5, lbl,
                fontsize=6, ha='center', va='bottom',
                color=ctrl_color, fontweight='bold', zorder=8,
                alpha=0.6,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])


# =====================================================================
# Run simulation
# =====================================================================

def run_sim(scenario_name, controller_key, seed=SEED):
    ctrl_cls = CONTROLLERS[controller_key][0]
    scenario = get_scenario(scenario_name, seed=seed)
    controller = ctrl_cls()
    sim = Simulator(scenario, controller, USVParameters())
    sim.run_to_completion()
    return sim.records, scenario, sim.get_metrics()


# =====================================================================
# Compute running COLREG compliance
# =====================================================================

def _step_is_compliant(rec):
    """Check if a single step is COLREG-compliant (mirrors metrics.py logic)."""
    enc = rec.encounter
    cmd = rec.command
    if enc is None or cmd is None:
        return None  # not applicable
    et = enc.encounter_type
    mn = cmd.manoeuvre
    if et not in ("head_on", "crossing_give_way", "overtaking"):
        return None  # not a give-way encounter
    if et == "head_on" and mn in ("early_stbd", "late_stbd", "slow_down"):
        return True
    if et == "crossing_give_way" and mn in ("early_stbd", "late_stbd", "slow_down", "stop"):
        return True
    if et == "overtaking" and mn in ("early_stbd", "late_stbd", "slow_down"):
        return True
    if mn == "hold_course" and rec.min_distance > getattr(cfg, 'AVOIDANCE_DOMAIN', 50.0):
        return True  # far enough, no action needed yet
    return False


def running_colreg(records, up_to):
    """Compute COLREG compliance from step 0 to up_to (inclusive)."""
    compliant_steps = 0
    total_steps = 0
    for i in range(up_to + 1):
        c = _step_is_compliant(records[i])
        if c is not None:
            total_steps += 1
            if c:
                compliant_steps += 1
    if total_steps == 0:
        return 1.0
    return compliant_steps / total_steps


# =====================================================================
# Frame renderer
# =====================================================================

def render_frame(fig, ax, records, scenario, metrics, frame_idx,
                 total_frames, scenario_name, controller_key):
    ax.clear()

    _, ctrl_display, ctrl_color, _ = CONTROLLERS[controller_key]

    rec = records[frame_idx]
    own = rec.state
    t = rec.time
    obs_list = rec.obstacles
    min_d = rec.min_distance

    # ── Axis limits (follow the action) ──
    all_x = [records[i].state.x for i in range(frame_idx + 1)]
    all_y = [records[i].state.y for i in range(frame_idx + 1)]
    for obs in obs_list:
        all_x.append(obs.x); all_y.append(obs.y)
    all_x.append(scenario.ownship_goal_x)
    all_y.append(scenario.ownship_goal_y)

    cx = (min(all_x) + max(all_x)) / 2.0
    cy = (min(all_y) + max(all_y)) / 2.0
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) + 50
    span = max(span, 80)
    half = span / 2.0

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, alpha=0.25, lw=0.5)

    # ── Goal ──
    ax.plot(scenario.ownship_goal_x, scenario.ownship_goal_y,
            '*', color=GOAL_COLOR, ms=16, markeredgecolor='#388E3C',
            markeredgewidth=1.0, zorder=8)
    ax.annotate("Goal", (scenario.ownship_goal_x, scenario.ownship_goal_y),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color='#388E3C', fontweight='bold', zorder=8)

    # ── Ownship trail (gradient) ──
    trail_x = [records[i].state.x for i in range(frame_idx + 1)]
    trail_y = [records[i].state.y for i in range(frame_idx + 1)]
    n_trail = len(trail_x)
    if n_trail > 2:
        for i in range(n_trail - 1):
            frac = i / max(n_trail - 1, 1)
            alpha = 0.15 + 0.55 * frac
            ax.plot(trail_x[i:i+2], trail_y[i:i+2],
                    color=ctrl_color, lw=2.0, alpha=alpha * 0.5, zorder=3)

    # ── Obstacle trails ──
    for oi, obs in enumerate(obs_list):
        if obs.is_static:
            continue
        ox = [records[i].obstacles[oi].x for i in range(frame_idx + 1)]
        oy = [records[i].obstacles[oi].y for i in range(frame_idx + 1)]
        n_ot = len(ox)
        if n_ot > 2:
            for i in range(n_ot - 1):
                frac = i / max(n_ot - 1, 1)
                a = 0.10 + 0.40 * frac
                ax.plot(ox[i:i+2], oy[i:i+2],
                        color=TARGET_TRAIL, lw=1.5, alpha=a, zorder=3)

    # ── Safety circles + D_safe ──
    own_r = math.sqrt(cfg.USV_LENGTH**2 + cfg.USV_WIDTH**2) / 2.0
    for obs in obs_list:
        obs_r = math.sqrt(obs.length**2 + obs.width**2) / 2.0
        sr = own_r + obs_r + getattr(cfg, 'VO_SAFETY_BUFFER', 2.0)
        draw_safety_circle(ax, obs.x, obs.y, sr)

    # ── Obstacle ships ──
    for obs in obs_list:
        oc = STATIC_COLOR if obs.is_static else TARGET_COLOR
        draw_ship(ax, obs.x, obs.y, obs.psi, obs.length, obs.width,
                  oc, alpha=0.85, fill=True, lw=1.5)
        ax.annotate(obs.label, (obs.x, obs.y),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=oc, fontweight='bold', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        if not obs.is_static:
            draw_heading_arrow(ax, obs.x, obs.y, obs.psi, obs.length,
                               oc, lw=1.2)

    # ── Distance line to closest obstacle ──
    closest = min(obs_list, key=lambda o: center_distance(own.x, own.y, o.x, o.y))
    ax.plot([own.x, closest.x], [own.y, closest.y],
            color='gray', ls=':', lw=0.8, alpha=0.6, zorder=2)
    mx, my = (own.x + closest.x) / 2, (own.y + closest.y) / 2
    d_color = "#F44336" if min_d < cfg.D_SAFE else "gray"
    ax.annotate(f"{min_d:.1f}m", (mx, my),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color=d_color, fontweight='bold', zorder=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=d_color,
                          alpha=0.8, lw=0.5))

    # ── CPA marker ──
    d_arr = np.array([records[i].min_distance for i in range(frame_idx + 1)])
    cpa_idx = int(np.argmin(d_arr))
    if frame_idx > cpa_idx + 5:
        cpax, cpay = records[cpa_idx].state.x, records[cpa_idx].state.y
        ax.plot(cpax, cpay, 'o', color='red', ms=8, zorder=9)
        ax.plot(cpax, cpay, 'o', color='lime', ms=4,
                markeredgecolor='green', markeredgewidth=0.8, zorder=10)
        ax.annotate(f"CPA\n{d_arr[cpa_idx]:.1f}m", (cpax, cpay),
                    textcoords="offset points", xytext=(-15, -15),
                    fontsize=7, color='red', fontweight='bold', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='red', alpha=0.8, lw=0.5))

    # ── Ownship ──
    draw_ship(ax, own.x, own.y, own.psi,
              cfg.USV_LENGTH, cfg.USV_WIDTH,
              ctrl_color, alpha=0.95, fill=True, lw=2.0, zorder=7)
    draw_heading_arrow(ax, own.x, own.y, own.psi,
                       cfg.USV_LENGTH, ctrl_color, lw=2.0, zorder=8)

    # ── Encounter classification (needed for overlays below) ──
    enc_type = rec.encounter.encounter_type if rec.encounter else "none"
    step_comp = _step_is_compliant(rec)
    manoeuvre = rec.command.manoeuvre if rec.command else "hold_course"

    # ==============================================================
    # Controller-specific overlays
    # ==============================================================
    sector_radius = cfg.AVOIDANCE_DOMAIN * 0.75

    if controller_key == "CC-CBF":
        # 1) COLREG sector wedges  2) Directional barrier contour  3) Rule badge
        draw_colreg_sectors(ax, own.x, own.y, own.psi,
                            sector_radius, enc_type)
        draw_cc_cbf_barrier(ax, own.x, own.y, own.psi, enc_type,
                            ctrl_color)
        draw_colreg_rule_badge(ax, enc_type, manoeuvre)

    elif controller_key == "C3BF":
        # Isotropic barrier circle
        draw_c3bf_barrier(ax, own.x, own.y, ctrl_color)

    elif controller_key == "Rule-COLREG":
        # COLREG sector wedges + rule badge
        draw_colreg_sectors(ax, own.x, own.y, own.psi,
                            sector_radius, enc_type)
        draw_colreg_rule_badge(ax, enc_type, manoeuvre)

    elif controller_key == "Geo-CRI":
        # CRI risk ring + gauge
        cri_val = rec.scalar_risk
        draw_cri_ring(ax, own.x, own.y, cri_val, cfg.AVOIDANCE_DOMAIN)
        draw_cri_gauge(ax, cri_val)

    elif controller_key == "TC-CBF":
        # Port & starboard turning circles
        draw_turning_circles(ax, own.x, own.y, own.psi, ctrl_color)

    # Running compliance
    colreg_now = running_colreg(records, frame_idx)

    # ── COLREG compliance indicator ──
    if step_comp is None:
        comp_label = "— N/A"
        comp_color = "#9E9E9E"
    elif step_comp:
        comp_label = "✓ Compliant"
        comp_color = "#4CAF50"
    else:
        comp_label = "✗ Violation"
        comp_color = "#F44336"

    # ── HUD ──
    speed_kn = own.u / 0.5144
    hud_lines = [
        f"t = {t:.1f}s",
        f"Speed: {own.u:.1f} m/s ({speed_kn:.1f} kn)",
        f"Min dist: {min_d:.1f} m",
        f"Encounter: {enc_type.replace('_', ' ').title()}",
        f"Manoeuvre: {manoeuvre.replace('_', ' ').title()}",
        f"COLREG now: {comp_label}",
        f"Compliance: {colreg_now:.2f}",
    ]
    if controller_key == "Geo-CRI":
        hud_lines.insert(3, f"CRI: {rec.scalar_risk:.3f}")
    hud = "\n".join(hud_lines)
    ax.text(0.02, 0.98, hud,
            transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='#BDBDBD', alpha=0.92, lw=0.8),
            zorder=12)

    # Compliance indicator dot
    ax.text(0.02, 0.58, "●", transform=ax.transAxes, fontsize=14,
            color=comp_color, va='top', zorder=12)

    # ── Metrics panel (bottom-left) ──
    # Show final metrics so far
    overall_min_sep = float(np.min(d_arr))
    panel = (
        f"Overall min sep: {overall_min_sep:.1f} m\n"
        f"Running COLREG: {colreg_now:.2f}"
    )
    panel_color = "#4CAF50" if overall_min_sep > 0 else "#F44336"
    ax.text(0.02, 0.02, panel,
            transform=ax.transAxes, fontsize=7.5,
            va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=panel_color, alpha=0.9, lw=0.8),
            zorder=12)

    # ── Title ──
    title_str = SCENARIO_TITLES.get(scenario_name, scenario_name)
    ax.set_title(f"{ctrl_display}  —  {title_str}",
                 fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # ── Legend ──
    legend_els = [
        Line2D([0], [0], color=ctrl_color, lw=2.5,
               label=f'Ownship ({ctrl_display})'),
        Line2D([0], [0], color=TARGET_COLOR, lw=2.0, label='Target Vessel'),
        Line2D([0], [0], color=SAFETY_COLOR, ls='--', lw=1.5,
               label='Safety Radius'),
        Line2D([0], [0], marker='o', color='red', markersize=6,
               markerfacecolor='lime', lw=0, label='CPA'),
    ]
    # Controller-specific legend items
    if controller_key == "CC-CBF":
        legend_els.append(
            Line2D([0], [0], color=ctrl_color, lw=1.5, alpha=0.55,
                   label=r'$R_{CC}$ barrier'))
        legend_els.append(
            Line2D([0], [0], color=ctrl_color, ls='--', lw=0.8, alpha=0.3,
                   label=r'$R_b$ (iso.)'))
    elif controller_key == "C3BF":
        legend_els.append(
            Line2D([0], [0], color=ctrl_color, ls='--', lw=1.2,
                   label=r'$R_{safe}$ barrier'))
    elif controller_key == "Geo-CRI":
        legend_els.append(
            Line2D([0], [0], color='#E65100', ls='-.', lw=1.5,
                   label='CRI threshold'))
    elif controller_key == "TC-CBF":
        legend_els.append(
            Line2D([0], [0], color=ctrl_color, ls='-', lw=1.2,
                   alpha=0.5, label='Turn circle (STBD)'))
        legend_els.append(
            Line2D([0], [0], color=ctrl_color, ls='--', lw=1.2,
                   alpha=0.5, label='Turn circle (PORT)'))

    ax.legend(handles=legend_els, loc='lower right', fontsize=6.5,
              framealpha=0.9, ncol=2)

    # ── Frame counter ──
    ax.text(0.5, -0.06, f"Frame {frame_idx+1}/{len(records)}",
            transform=ax.transAxes, fontsize=7, ha='center', color='gray')


# =====================================================================
# GIF generator
# =====================================================================

def generate_gif(scenario_name, controller_key, output_dir=OUTPUT_DIR):
    _, ctrl_display, _, ctrl_suffix = CONTROLLERS[controller_key]

    print(f"\n  Generating: {scenario_name} / {ctrl_display}")

    t0 = time.time()
    records, scenario, metrics = run_sim(scenario_name, controller_key)
    n_steps = len(records)
    print(f"    Sim: {n_steps} steps, {records[-1].time:.1f}s | "
          f"MinSep={metrics.min_separation:.1f}m  COLREG={metrics.colreg_compliance:.2f}")

    # Frame selection
    indices = list(range(0, n_steps, FRAME_SKIP))
    if indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)

    max_frames = GIF_DURATION_CAP * FPS
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(int(max_frames))]
        if indices[-1] != n_steps - 1:
            indices.append(n_steps - 1)

    total = len(indices)
    print(f"    Rendering {total} frames @ {FPS} fps...")

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout(pad=2.0)

    images = []
    for fi, fidx in enumerate(indices):
        render_frame(fig, ax, records, scenario, metrics,
                     fidx, total, scenario_name, controller_key)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

        if (fi + 1) % 30 == 0 or fi == total - 1:
            print(f"      Frame {fi+1}/{total}")

    plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir,
                            f"{scenario_name}_{ctrl_suffix}.gif")

    # Pause at end (1 second)
    for _ in range(FPS):
        images.append(images[-1])

    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)

    sz = os.path.getsize(gif_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"    ✓ {os.path.basename(gif_path)}  ({sz:.1f} MB, {elapsed:.1f}s)")
    return gif_path


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper GIFs")
    parser.add_argument("--controller", type=str, default=None,
                        help="Single controller key (CC-CBF, C3BF, ...)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Single scenario (head_on, ...)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Paper GIF Generator  (5 controllers × 4 scenarios)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    ctrls = [args.controller] if args.controller else list(CONTROLLERS.keys())
    scens = [args.scenario] if args.scenario else SCENARIOS

    paths = []
    for ck in ctrls:
        for sn in scens:
            p = generate_gif(sn, ck)
            paths.append(p)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Done! {len(paths)} GIFs in {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    for p in paths:
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"    {os.path.basename(p):45s} {sz:.1f} MB")


if __name__ == "__main__":
    main()
