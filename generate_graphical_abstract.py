#!/usr/bin/env python3
"""
generate_graphical_abstract.py
===============================
Journal graphical abstract – 2 × 2 grid, one COLREGS scenario per panel.

Layout:
  ┌──────────────────┬──────────────────┐
  │  Head-On  R.14   │  Crossing  R.15  │
  ├──────────────────┼──────────────────┤
  │  Overtaking R.13 │  Static obs.     │
  └──────────────────┴──────────────────┘

Each panel shows:
  • Sea background with subtle grid
  • Faded full trajectory (own-ship blue, obstacle red)
  • Vivid avoidance manoeuvre segment
  • Ship-hull icons at CPA
  • CC-CBF barrier drawn at every non-overlapping step along the path
    (faint when far from obstacle, vivid near CPA)
  • Small polar inset (paper barrier_shape style): filled lobe, dashed R_b,
    θ_C line, USV icon — positioned upper-right of panel
  • Goal star, safety circle, scale bar, title & COLREGS rule badge

Output: paper/figures/graphical_abstract.png
"""

import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# ── Global font scaling (makes figure readable in single-column print) ────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        16,        # base size — all text inherits this
    "axes.titlesize":   17,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
    "figure.titlesize": 18,
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.scenario import get_scenario
from core.simulator import Simulator
from core.entities import USVParameters
from algorithms.cc_cbf import CCCBFController, R_BASE, LAMBDA, THETA_C, ACTIVATION_RANGE, _obs_rb
import data.config as cfg

# ─────────────────────────────────────────────────────────────────────────────
#  Barrier activity helper
# ─────────────────────────────────────────────────────────────────────────────
# Scale at which h_cc = this value → lam_draw = 0 (plain circle).
# h_cc = dist² - R_cc² so at dist>>R_base, h >> 0 and the barrier is slack.
# We ramp lam linearly from 0 (h >= H_SLACK) down to full lam (h <= R_BASE²).
_H_FULL  = R_BASE * R_BASE          # ≈ 161 m² — on the safety boundary
_H_SLACK = 20.0 * R_BASE * R_BASE   # ≈ 3222 m² — barrier completely slack

def _effective_lam(h_val: float, lam: float) -> float:
    """
    Return an effective lambda that is 0 when the CBF is slack (h >> R²)
    and ramps to full lam as h → R_BASE².
    This makes the inflated lobe appear only when the barrier is actually
    constraining the ownship, matching the animated GIF behaviour.
    """
    if h_val >= _H_SLACK:
        return 0.0
    if h_val <= _H_FULL:
        return lam
    # Linear ramp between _H_SLACK and _H_FULL
    t = (h_val - _H_FULL) / (_H_SLACK - _H_FULL)   # 1 when slack, 0 when tight
    return lam * (1.0 - t)


def _h_cc_at(rec, lam_val: float) -> float:
    """
    Compute the CC-CBF value h = dist² - R_cc² for the first obstacle
    in a simulation record.  Returns a large sentinel (H_SLACK) when
    no obstacle is present.
    """
    if not rec.obstacles:
        return _H_SLACK
    o   = rec.obstacles[0]
    ox, oy, opsi = rec.state.x, rec.state.y, rec.state.psi
    tx, ty       = o.x, o.y
    dx, dy       = tx - ox, ty - oy
    d2           = dx * dx + dy * dy
    rb           = _obs_rb(o.length, o.width)
    theta        = math.atan2(dy, dx) - opsi
    # Wrap to [-pi, pi]
    theta        = (theta + math.pi) % (2 * math.pi) - math.pi
    cos_t        = max(0.0, math.cos(theta))
    rcc          = rb * (1.0 + lam_val * cos_t * cos_t)
    return d2 - rcc * rcc

# ─────────────────────────────────────────────────────────────────────────────
#  Global constants
# ─────────────────────────────────────────────────────────────────────────────
OUT_PATH = "paper/figures/graphical_abstract.png"
DPI      = 350

C_OWN  = "#1565C0"   # ownship blue
C_OBS  = "#C62828"   # obstacle red
C_SAFE = "#2E7D32"   # safety circle green
C_GOAL = "#00C853"   # goal star
C_SEA  = "#FFFFFF"   # sea background (white)
C_GRID = "#E0E0E0"   # grid lines

# Scenarios: (sim_name, display_title, COLREGS_rule, barrier_colour, λ, θ_c)
# Original colours (current figures)
SCENARIOS = [
    ("overtaking", "Overtaking", "Rule 13", "#00695C", 0.45, math.radians( 30)),
    ("head_on",    "Head-On",    "Rule 14", "#1565C0", 0.55, math.radians(-45)),
]

# Barrier-shape colours — matching paper/figures/barrier_shape.png exactly
#   Head-On  → #1565C0,  Overtaking → #FB8C00
SCENARIOS_BARRIER_STYLE = [
    ("overtaking", "Overtaking", "Rule 13", "#FB8C00", 0.45, math.radians( 30)),
    ("head_on",    "Head-On",    "Rule 14", "#1565C0", 0.55, math.radians(-45)),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate(scenario_name):
    sc  = get_scenario(scenario_name, seed=42)
    sim = Simulator(sc, CCCBFController(), USVParameters())
    sim.run_to_completion()
    return sc, sim.records


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def view_window(records, scenario, pad=None, cell_aspect=2.4):
    """
    Fit the full trajectory of both vessels with padding, then expand
    whichever axis is too small so the data fills the cell at equal aspect.

    pad defaults to R_BASE * 1.7 so the first and last barrier lobes
    (radius ≈ R_BASE * (1+λ) ≤ R_BASE * 1.6) never get clipped.
    """
    if pad is None:
        pad = R_BASE * 1.7   # ~21 m — safely larger than any barrier lobe

    xs, ys = [], []
    for r in records:
        xs.append(r.state.x); ys.append(r.state.y)
        for o in r.obstacles:
            xs.append(o.x); ys.append(o.y)
    xs.append(scenario.ownship_goal_x)
    ys.append(scenario.ownship_goal_y)

    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    data_w = xmax - xmin
    data_h = ymax - ymin
    data_aspect = data_w / max(data_h, 1e-6)

    if data_aspect > cell_aspect:
        # Data is wider than the cell → expand y to match
        target_h = data_w / cell_aspect
        yc = (ymin + ymax) / 2
        ymin = yc - target_h / 2
        ymax = yc + target_h / 2
    else:
        # Data is taller than the cell → expand x to match
        target_w = data_h * cell_aspect
        xc = (xmin + xmax) / 2
        xmin = xc - target_w / 2
        xmax = xc + target_w / 2

    return (xmin, xmax), (ymin, ymax)


def cbf_contour(ox, oy, psi_own, lam, theta_c, n_pts=180):
    """CC-CBF barrier contour (closed polygon) centred at (ox, oy)."""
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    xs, ys = [], []
    for a in angles:
        phi = max(0.0, math.cos((a - psi_own) - theta_c))
        r   = R_BASE * (1.0 + lam * phi)
        xs.append(ox + r * math.cos(a))
        ys.append(oy + r * math.sin(a))
    xs.append(xs[0]); ys.append(ys[0])
    return np.array(xs), np.array(ys)


def hull_polygon(cx, cy, psi, length, width):
    """Simple ship-hull polygon centred at (cx, cy)."""
    pts = np.array([
        [ length * 0.55,  0           ],
        [ length * 0.20,  width * 0.50],
        [-length * 0.45,  width * 0.40],
        [-length * 0.45, -width * 0.40],
        [ length * 0.20, -width * 0.50],
    ])
    c, s = math.cos(psi), math.sin(psi)
    R    = np.array([[c, -s], [s, c]])
    rot  = pts @ R.T
    return rot[:, 0] + cx, rot[:, 1] + cy


def gradient_trail(ax, xs, ys, color, lw=2.0,
                   alpha_start=0.08, alpha_end=0.55):
    """Line collection that fades from transparent (old) → opaque (recent)."""
    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n    = len(segs)
    r, g, b, _ = mcolors.to_rgba(color)
    colors = [(r, g, b, a) for a in np.linspace(alpha_start, alpha_end, n)]
    lc = LineCollection(segs, colors=colors, linewidths=lw, zorder=4)
    ax.add_collection(lc)


# ─────────────────────────────────────────────────────────────────────────────
#  Barrier evolution along the path  (non-overlapping)
# ─────────────────────────────────────────────────────────────────────────────
def draw_barrier_evolution(ax, records, lam, theta_c, accent, bshape=False,
                           extra_inflated_slots=None):
    """
    Walk the trajectory and place a CC-CBF barrier snapshot every time the
    ownship has moved >= 2 * R_BASE from the last placed barrier centre.

    bshape=False  → proximity-scaled opacity (default graphical-abstract style)
    bshape=True   → flat alpha=0.10 / lw=1.8, matching paper/figures/barrier_shape.png

    extra_inflated_slots: optional set of 0-based slot indices within
        placed_frames whose barriers should be drawn with a forced moderate
        tightness (0.45) to show an inflated lobe even if h_cc is large.
        Used to add two inflated visualisations immediately before and after
        the naturally-tight region (head-on scenario).
    """
    r_max   = R_BASE * (1.0 + lam)
    min_gap = 2.0 * R_BASE          # ~25 m — dense but non-overlapping
    _extra_slots = set(extra_inflated_slots) if extra_inflated_slots else set()

    placed_centres = []
    placed_frames  = []

    for i, rec in enumerate(records):
        cx, cy = rec.state.x, rec.state.y
        if placed_centres:
            px, py = placed_centres[-1]
            if math.hypot(cx - px, cy - py) < min_gap:
                continue
        placed_centres.append((cx, cy))
        placed_frames.append(i)

    if not placed_frames:
        return

    dists = []
    for i in placed_frames:
        rec = records[i]
        if rec.obstacles:
            o = rec.obstacles[0]
            dists.append(math.hypot(o.x - rec.state.x, o.y - rec.state.y))
        else:
            dists.append(999.0)

    for slot, (frame_idx, dist) in enumerate(zip(placed_frames, dists)):
        rec = records[frame_idx]

        # Physics-accurate: use the actual h_cc value to determine how much
        # the barrier lobe is inflated.  When h is large (ownship well inside
        # the safe set, no constraint active) lam_draw → 0 (plain circle).
        # As h shrinks toward R_BASE² (boundary) lam_draw → full lam.
        h_val    = _h_cc_at(rec, lam)
        lam_draw = _effective_lam(h_val, lam)
        # Normalised tightness: 0 = slack, 1 = on boundary
        tightness = 1.0 - max(0.0, min(1.0, (h_val - _H_FULL) / (_H_SLACK - _H_FULL)))

        # For explicitly requested extra-inflated slots, force a moderate
        # tightness so the lobe is visible even though h_cc is large.
        if slot in _extra_slots and tightness < 0.45:
            tightness = 0.45
            lam_draw  = lam * 0.45

        # When extra_inflated_slots is active (head-on), all plain circles
        # should share the same colour weight as the inflated barriers —
        # use a minimum style tightness of 0.45 for alpha/lw computation
        # but do NOT change lam_draw (lobe shape stays a plain circle).
        style_tightness = tightness
        if _extra_slots and style_tightness < 0.45:
            style_tightness = 0.45

        bx, by = cbf_contour(rec.state.x, rec.state.y,
                              rec.state.psi, lam_draw, theta_c)

        if bshape:
            if style_tightness > 0.01:
                fill_alpha = 0.04 + 0.08 * style_tightness   # up to 0.12 when tight
                edge_alpha = 0.30 + 0.70 * style_tightness   # up to 1.0 when tight
                edge_lw    = 1.0 + 0.8 * style_tightness     # up to 1.8 when tight
            else:
                fill_alpha = 0.03
                edge_alpha = 0.20
                edge_lw    = 0.8
        else:
            proximity  = math.exp(-dist / (1.8 * r_max))
            fill_alpha = 0.04 + 0.46 * style_tightness * (0.2 + 0.8 * proximity)
            edge_alpha = 0.20 + 0.75 * style_tightness * (0.5 + 0.5 * proximity)
            edge_lw    = 1.4

        ax.fill(bx, by, color=accent, alpha=fill_alpha, zorder=2)
        ax.plot(bx, by, color=accent, alpha=edge_alpha,
                lw=edge_lw, zorder=3)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-panel renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_panel(fig, ax, sc, records, title, rule, accent, lam, theta_c,
                 bshape=False, show_obs_end=True, extra_inflated_slots=None):
    xlim, ylim = view_window(records, sc)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    # ── Axes setup ────────────────────────────────────────────────────────
    ax.set_facecolor(C_SEA)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#AAAAAA")

    # Light grid
    for x in np.arange(math.floor(xlim[0] / 20) * 20, xlim[1] + 20, 20):
        ax.axvline(x, color=C_GRID, lw=0.4, zorder=0)
    for y in np.arange(math.floor(ylim[0] / 20) * 20, ylim[1] + 20, 20):
        ax.axhline(y, color=C_GRID, lw=0.4, zorder=0)

    # ── Trajectories ──────────────────────────────────────────────────────
    own_x = [r.state.x for r in records]
    own_y = [r.state.y for r in records]
    obs_x = [r.obstacles[0].x for r in records if r.obstacles]
    obs_y = [r.obstacles[0].y for r in records if r.obstacles]

    gradient_trail(ax, own_x, own_y, C_OWN,
                   lw=3.0, alpha_start=0.10, alpha_end=0.80)
    if obs_x:
        gradient_trail(ax, obs_x, obs_y, C_OBS,
                       lw=3.0, alpha_start=0.10, alpha_end=0.80)

    # ── CC-CBF barrier evolution along the path ───────────────────────────
    draw_barrier_evolution(ax, records, lam, theta_c, accent, bshape=bshape,
                           extra_inflated_slots=extra_inflated_slots)

    # ── CPA frame ─────────────────────────────────────────────────────────
    cpa_idx = int(np.argmin([r.min_distance for r in records]))
    cpa_rec = records[cpa_idx]

    # Safety circle at CPA obstacle position
    if cpa_rec.obstacles:
        ox, oy = cpa_rec.obstacles[0].x, cpa_rec.obstacles[0].y
        ax.add_patch(Circle((ox, oy), cfg.D_SAFE,
                            fill=False, ec=C_SAFE, lw=2.0,
                            ls=":", zorder=5, alpha=0.75))

    # ── Ship hulls at START position (ghost) ──────────────────────────────
    start_rec = records[0]
    hx_s, hy_s = hull_polygon(start_rec.state.x, start_rec.state.y,
                               start_rec.state.psi,
                               cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx_s, hy_s, color=C_OWN, zorder=6, alpha=0.30)
    ax.plot(hx_s, hy_s, color=C_OWN, lw=0.8,  zorder=6, alpha=0.55)

    if start_rec.obstacles:
        os_     = start_rec.obstacles[0]
        psi_os  = math.atan2(os_.vy, os_.vx) if (abs(os_.vx) + abs(os_.vy)) > 0.1 \
                  else 0.0
        hx_os, hy_os = hull_polygon(os_.x, os_.y, psi_os,
                                     cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx_os, hy_os, color=C_OBS, zorder=6, alpha=0.30)
        ax.plot(hx_os, hy_os, color=C_OBS, lw=0.8,  zorder=6, alpha=0.55)

    # ── Ship hulls at CPA (solid) ─────────────────────────────────────────
    hx, hy = hull_polygon(cpa_rec.state.x, cpa_rec.state.y,
                           cpa_rec.state.psi,
                           cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx, hy, color=C_OWN,  zorder=8, alpha=0.95)
    ax.plot(hx, hy, color="white", lw=1.2, zorder=9)

    if cpa_rec.obstacles:
        o      = cpa_rec.obstacles[0]
        psi_ob = math.atan2(o.vy, o.vx) if (abs(o.vx) + abs(o.vy)) > 0.1 \
                 else 0.0
        hx2, hy2 = hull_polygon(o.x, o.y, psi_ob,
                                  cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx2, hy2, color=C_OBS,  zorder=8, alpha=0.95)
        ax.plot(hx2, hy2, color="white", lw=1.2, zorder=9)

    # ── Ship hulls at end position ─────────────────────────────────────────
    end_rec = records[-1]
    hx_e, hy_e = hull_polygon(end_rec.state.x, end_rec.state.y,
                               end_rec.state.psi,
                               cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx_e, hy_e, color=C_OWN,  zorder=7, alpha=0.45)
    ax.plot(hx_e, hy_e, color=C_OWN,  lw=0.8,   zorder=7, alpha=0.70)

    if end_rec.obstacles and show_obs_end:
        oe     = end_rec.obstacles[0]
        psi_oe = math.atan2(oe.vy, oe.vx) if (abs(oe.vx) + abs(oe.vy)) > 0.1 \
                 else 0.0
        hx_oe, hy_oe = hull_polygon(oe.x, oe.y, psi_oe,
                                     cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx_oe, hy_oe, color=C_OBS, zorder=7, alpha=0.45)
        ax.plot(hx_oe, hy_oe, color=C_OBS, lw=0.8,  zorder=7, alpha=0.70)

    # ── Direction arrows (near CPA) ───────────────────────────────────────
    n    = len(records)
    mid  = max(0, cpa_idx - n // 10)
    step = max(1, n // 50)
    if mid + step < len(own_x):
        ax.annotate("", xy=(own_x[mid + step], own_y[mid + step]),
                    xytext=(own_x[mid], own_y[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OWN,
                                   lw=1.5, mutation_scale=18), zorder=10)
    if obs_x and mid + step < len(obs_x):
        ax.annotate("", xy=(obs_x[mid + step], obs_y[mid + step]),
                    xytext=(obs_x[mid],  obs_y[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OBS,
                                   lw=1.5, mutation_scale=18), zorder=10)

    # ── Text labels with boxes ────────────────────────────────────────────
    tx = xlim[0] + xspan * 0.03
    ty = ylim[1] - yspan * 0.03
    ax.text(tx, ty, title, ha="left", va="top",
            fontsize=17, fontweight="bold", color="#0D2D6B",
            zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#0D2D6B", linewidth=2.0, alpha=0.92))
    ax.text(xlim[1] - xspan * 0.03, ty, rule,
            ha="right", va="top", fontsize=20, fontstyle="italic",
            color=accent, zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=accent, linewidth=2.0, alpha=0.92))


# ─────────────────────────────────────────────────────────────────────────────
#  COLREG sector wedge helper  (ported from generate_paper_gifs.py)
# ─────────────────────────────────────────────────────────────────────────────
from matplotlib.patches import Wedge
import matplotlib.patheffects as _pe   # alias – pe already imported at top

# (name, start_offset_deg, end_offset_deg, colour) — offsets from heading
_SECTOR_DEFS = [
    ("Head-On",           -15,     15,    "#FF9800"),
    ("Crossing\nGive-Way",-112.5, -15,   "#EF5350"),
    ("Crossing\nStand-On",  15,  112.5,  "#66BB6A"),
    ("Overtaking",         112.5, 247.5, "#42A5F5"),
]

# Map display name → COLREG record encounter_type strings
_ENC_NAME_MAP = {
    "overtaking": "Overtaking",
    "head_on":    "Head-On",
    "crossing_give_way": "Crossing\nGive-Way",
    "crossing_stand_on": "Crossing\nStand-On",
}


def _draw_colreg_sectors_at(ax, cx, cy, psi, radius, active_sector):
    """
    Draw all four COLREG sector wedges centred at (cx, cy).
    active_sector: display-name string matching _SECTOR_DEFS key, or None.
    """
    heading_deg = math.degrees(psi)
    for sec_name, s_off, e_off, clr in _SECTOR_DEFS:
        active = (sec_name == active_sector)
        alpha  = 0.28 if active else 0.07
        lw     = 1.8  if active else 0.5
        theta1 = heading_deg + s_off
        theta2 = heading_deg + e_off
        w = Wedge((cx, cy), radius, theta1, theta2,
                  facecolor=clr, edgecolor=clr,
                  alpha=alpha, lw=lw, zorder=1)
        ax.add_patch(w)
        if active:
            mid_a = math.radians((theta1 + theta2) / 2.0)
            lx = cx + radius * 0.52 * math.cos(mid_a)
            ly = cy + radius * 0.52 * math.sin(mid_a)
            ax.text(lx, ly, sec_name, fontsize=17, ha="center", va="center",
                    color=clr, fontweight="bold", zorder=9,
                    path_effects=[_pe.withStroke(linewidth=2,
                                                 foreground="white")])


def render_panel_colreg(fig, ax, sc, records, scenario_name,
                        title, rule, accent, lam, theta_c):
    """
    Like render_panel but also overlays COLREG sector wedges + active sector
    highlight at every CC-CBF barrier snapshot, so readers see the spatial
    relationship between encounter geometry and the barrier lobe — exactly
    as visible in the animated GIFs.
    """
    from matplotlib.patches import FancyArrowPatch

    xlim, ylim = view_window(records, sc)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_facecolor(C_SEA)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#AAAAAA")

    # Light grid
    for x in np.arange(math.floor(xlim[0] / 20) * 20, xlim[1] + 20, 20):
        ax.axvline(x, color=C_GRID, lw=0.4, zorder=0)
    for y in np.arange(math.floor(ylim[0] / 20) * 20, ylim[1] + 20, 20):
        ax.axhline(y, color=C_GRID, lw=0.4, zorder=0)

    # ── Trajectories ─────────────────────────────────────────────────────
    own_x = [r.state.x for r in records]
    own_y = [r.state.y for r in records]
    obs_x = [r.obstacles[0].x for r in records if r.obstacles]
    obs_y = [r.obstacles[0].y for r in records if r.obstacles]

    gradient_trail(ax, own_x, own_y, C_OWN,
                   lw=3.0, alpha_start=0.10, alpha_end=0.80)
    if obs_x:
        gradient_trail(ax, obs_x, obs_y, C_OBS,
                       lw=3.0, alpha_start=0.10, alpha_end=0.80)

    # ── COLREG sectors + CC-CBF barrier at every snapshot step ───────────
    r_max   = R_BASE * (1.0 + lam)
    min_gap = 2.0 * R_BASE
    sector_radius = r_max * 1.55   # wedges slightly larger than the lobe

    # Active sector name for this scenario (constant throughout encounter)
    active_sector = _ENC_NAME_MAP.get(scenario_name, None)

    placed_centres = []
    placed_frames  = []
    for i, rec in enumerate(records):
        cx, cy = rec.state.x, rec.state.y
        if placed_centres:
            px, py = placed_centres[-1]
            if math.hypot(cx - px, cy - py) < min_gap:
                continue
        placed_centres.append((cx, cy))
        placed_frames.append(i)

    dists = []
    for i in placed_frames:
        rec = records[i]
        if rec.obstacles:
            o = rec.obstacles[0]
            dists.append(math.hypot(o.x - rec.state.x, o.y - rec.state.y))
        else:
            dists.append(999.0)

    for frame_idx, dist in zip(placed_frames, dists):
        rec = records[frame_idx]
        cx, cy, psi = rec.state.x, rec.state.y, rec.state.psi

        # 1) COLREG sector wedges (all four; active one highlighted)
        _draw_colreg_sectors_at(ax, cx, cy, psi, sector_radius, active_sector)

        # 2) CC-CBF barrier contour — use actual h_cc to scale the lobe
        h_val     = _h_cc_at(rec, lam)
        lam_draw  = _effective_lam(h_val, lam)
        tightness = 1.0 - max(0.0, min(1.0, (h_val - _H_FULL) / (_H_SLACK - _H_FULL)))
        bx, by = cbf_contour(cx, cy, psi, lam_draw, theta_c)
        proximity  = math.exp(-dist / (1.8 * r_max))
        fill_alpha = 0.04 + 0.18 * tightness * (0.5 + 0.5 * proximity)
        edge_alpha = 0.20 + 0.80 * tightness * (0.5 + 0.5 * proximity)
        ax.fill(bx, by, color=accent, alpha=fill_alpha, zorder=3)
        ax.plot(bx, by, color=accent, alpha=edge_alpha, lw=1.6, zorder=4)

    # ── CPA frame ─────────────────────────────────────────────────────────
    cpa_idx = int(np.argmin([r.min_distance for r in records]))
    cpa_rec = records[cpa_idx]

    # Safety circle
    if cpa_rec.obstacles:
        ox, oy = cpa_rec.obstacles[0].x, cpa_rec.obstacles[0].y
        ax.add_patch(Circle((ox, oy), cfg.D_SAFE,
                            fill=False, ec=C_SAFE, lw=2.0,
                            ls=":", zorder=5, alpha=0.75))

    # ── Ship hulls at START position (ghost) ──────────────────────────────
    start_rec = records[0]
    hx_s, hy_s = hull_polygon(start_rec.state.x, start_rec.state.y,
                               start_rec.state.psi,
                               cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx_s, hy_s, color=C_OWN, zorder=6, alpha=0.30)
    ax.plot(hx_s, hy_s, color=C_OWN, lw=0.8,  zorder=6, alpha=0.55)

    if start_rec.obstacles:
        os_     = start_rec.obstacles[0]
        psi_os  = math.atan2(os_.vy, os_.vx) if (abs(os_.vx) + abs(os_.vy)) > 0.1 \
                  else 0.0
        hx_os, hy_os = hull_polygon(os_.x, os_.y, psi_os,
                                     cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx_os, hy_os, color=C_OBS, zorder=6, alpha=0.30)
        ax.plot(hx_os, hy_os, color=C_OBS, lw=0.8,  zorder=6, alpha=0.55)

    # ── Ship hulls at CPA (solid) ─────────────────────────────────────────
    hx, hy = hull_polygon(cpa_rec.state.x, cpa_rec.state.y,
                           cpa_rec.state.psi,
                           cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx, hy, color=C_OWN,  zorder=8, alpha=0.95)
    ax.plot(hx, hy, color="white", lw=1.2, zorder=9)

    if cpa_rec.obstacles:
        o      = cpa_rec.obstacles[0]
        psi_ob = math.atan2(o.vy, o.vx) if (abs(o.vx) + abs(o.vy)) > 0.1 \
                 else 0.0
        hx2, hy2 = hull_polygon(o.x, o.y, psi_ob,
                                  cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx2, hy2, color=C_OBS,  zorder=8, alpha=0.95)
        ax.plot(hx2, hy2, color="white", lw=1.2, zorder=9)

    # ── Ship hulls at end position (ghost) ───────────────────────────────
    end_rec = records[-1]
    hx_e, hy_e = hull_polygon(end_rec.state.x, end_rec.state.y,
                               end_rec.state.psi,
                               cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx_e, hy_e, color=C_OWN, zorder=7, alpha=0.45)
    ax.plot(hx_e, hy_e, color=C_OWN, lw=0.8,  zorder=7, alpha=0.70)

    if end_rec.obstacles:
        oe     = end_rec.obstacles[0]
        psi_oe = math.atan2(oe.vy, oe.vx) if (abs(oe.vx) + abs(oe.vy)) > 0.1 \
                 else 0.0
        hx_oe, hy_oe = hull_polygon(oe.x, oe.y, psi_oe,
                                     cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx_oe, hy_oe, color=C_OBS, zorder=7, alpha=0.45)
        ax.plot(hx_oe, hy_oe, color=C_OBS, lw=0.8,  zorder=7, alpha=0.70)

    # ── Direction arrows near CPA ─────────────────────────────────────────
    n    = len(records)
    mid  = max(0, cpa_idx - n // 10)
    step = max(1, n // 50)
    if mid + step < len(own_x):
        ax.annotate("", xy=(own_x[mid + step], own_y[mid + step]),
                    xytext=(own_x[mid], own_y[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OWN,
                                   lw=1.5, mutation_scale=18), zorder=10)
    if obs_x and mid + step < len(obs_x):
        ax.annotate("", xy=(obs_x[mid + step], obs_y[mid + step]),
                    xytext=(obs_x[mid],  obs_y[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OBS,
                                   lw=1.5, mutation_scale=18), zorder=10)

    # ── Sector legend (bottom-right, inside panel) ────────────────────────
    # Small coloured squares + labels for all four sectors
    bx0 = xlim[1] - xspan * 0.01
    by0 = ylim[0] + yspan * 0.02
    row_h = yspan * 0.055
    for k, (sec_name, _, _, clr) in enumerate(_SECTOR_DEFS):
        bx_r = bx0 - xspan * 0.025
        by_r = by0 + k * row_h
        ax.add_patch(plt.Rectangle(
            (bx_r, by_r), xspan * 0.022, row_h * 0.65,
            color=clr, alpha=0.55, zorder=12, lw=0
        ))
        label_text = sec_name.replace("\n", " ")
        ax.text(bx_r - xspan * 0.005, by_r + row_h * 0.30,
                label_text, ha="right", va="center",
                fontsize=17, color=clr, fontweight="bold", zorder=13,
                path_effects=[_pe.withStroke(linewidth=1.5,
                                             foreground="white")])

    # ── Boxed title + rule labels ─────────────────────────────────────────
    tx = xlim[0] + xspan * 0.03
    ty = ylim[1] - yspan * 0.03
    ax.text(tx, ty, title, ha="left", va="top",
            fontsize=17, fontweight="bold", color="#0D2D6B",
            zorder=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#0D2D6B", linewidth=2.0, alpha=0.92))
    ax.text(xlim[1] - xspan * 0.03, ty, rule,
            ha="right", va="top", fontsize=20, fontstyle="italic",
            color=accent, zorder=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=accent, linewidth=2.0, alpha=0.92))


def add_legend_colreg(fig):
    """Extended legend: trajectory lines + COLREG sector colour key."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    handles = [
        mlines.Line2D([], [], color=C_OWN,  lw=2.2, label="USV path"),
        mlines.Line2D([], [], color=C_OBS,  lw=2.2, label="Obstacle path"),
        mlines.Line2D([], [], color=C_SAFE, lw=1.2, ls=":",
                      label="Safety circle"),
    ]
    for sec_name, _, _, clr in _SECTOR_DEFS:
        handles.append(mpatches.Patch(
            facecolor=clr, alpha=0.55,
            label=sec_name.replace("\n", " ") + " sector",
        ))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=20, frameon=True, framealpha=0.92,
               edgecolor="#AAAAAA", bbox_to_anchor=(0.5, 0.00))


# ─────────────────────────────────────────────────────────────────────────────
#  Active-rule-only panel  (Figure 6)
# ─────────────────────────────────────────────────────────────────────────────

# Sector definitions keyed by raw encounter_type string from records
_ACTIVE_SECTOR = {
    "head_on":            ("-15°  to +15°",   "#FF9800", "Rule 14 · Head-On",           -15,     15   ),
    "overtaking":         ("±112.5° to ±180°","#42A5F5", "Rule 13 · Overtaking",         112.5,  247.5),
    "crossing_give_way":  ("-112.5° to -15°", "#EF5350", "Rule 15 · Crossing Give-Way", -112.5,  -15  ),
    "crossing_stand_on":  ("+15° to +112.5°", "#66BB6A", "Rule 17 · Crossing Stand-On",  15,     112.5),
}


def _draw_active_wedge(ax, cx, cy, psi, radius, enc_type):
    """
    Draw ONLY the wedge for the active COLREG encounter type.
    Returns the sector colour (or None if enc_type not recognised).
    """
    if enc_type not in _ACTIVE_SECTOR:
        return None
    _, clr, label, s_off, e_off = _ACTIVE_SECTOR[enc_type]
    heading_deg = math.degrees(psi)
    theta1 = heading_deg + s_off
    theta2 = heading_deg + e_off
    w = Wedge((cx, cy), radius, theta1, theta2,
              facecolor=clr, edgecolor=clr,
              alpha=0.30, lw=2.0, zorder=1)
    ax.add_patch(w)
    # Label inside wedge
    mid_a = math.radians((theta1 + theta2) / 2.0)
    lx = cx + radius * 0.55 * math.cos(mid_a)
    ly = cy + radius * 0.55 * math.sin(mid_a)
    short = label.split("·")[1].strip()   # e.g. "Head-On"
    ax.text(lx, ly, short, fontsize=17, ha="center", va="center",
            color=clr, fontweight="bold", zorder=9,
            path_effects=[_pe.withStroke(linewidth=2, foreground="white")])
    return clr


def render_panel_active_rule(fig, ax, sc, records, scenario_name,
                              title, rule, accent, lam, theta_c,
                              force_enc=None):
    """
    Shows the ACTIVE COLREG sector wedge continuously along the entire path
    (dense tiling, one wedge every ~sector_radius step) together with the
    CC-CBF barrier lobe at non-overlapping barrier intervals.

    force_enc: if given, always use this encounter_type key (e.g. "overtaking"
               or "head_on"), ignoring whatever the records say.  This gives a
               clean, stable sector that never flickers.
    """
    xlim, ylim = view_window(records, sc)
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_facecolor(C_SEA)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#AAAAAA")

    # Light grid
    for x in np.arange(math.floor(xlim[0] / 20) * 20, xlim[1] + 20, 20):
        ax.axvline(x, color=C_GRID, lw=0.4, zorder=0)
    for y in np.arange(math.floor(ylim[0] / 20) * 20, ylim[1] + 20, 20):
        ax.axhline(y, color=C_GRID, lw=0.4, zorder=0)

    # ── Trajectories ─────────────────────────────────────────────────────
    own_x = [r.state.x for r in records]
    own_y = [r.state.y for r in records]
    obs_x = [r.obstacles[0].x for r in records if r.obstacles]
    obs_y = [r.obstacles[0].y for r in records if r.obstacles]

    gradient_trail(ax, own_x, own_y, C_OWN,
                   lw=3.0, alpha_start=0.10, alpha_end=0.80)
    if obs_x:
        gradient_trail(ax, obs_x, obs_y, C_OBS,
                       lw=3.0, alpha_start=0.10, alpha_end=0.80)

    r_max         = R_BASE * (1.0 + lam)
    sector_radius = r_max * 1.55
    min_gap       = 2.0 * R_BASE          # barrier snapshot spacing
    wedge_gap     = sector_radius * 1.0   # wedge tiling spacing (full coverage)

    # ── Pass 1: dense wedge tiling along the full path ────────────────────
    # Walk every step; place a wedge whenever the USV has moved >= wedge_gap
    # from the last wedge centre.  Uses a very low alpha so the overlapping
    # wedges build up a soft "filled lane" without being opaque individually.
    wedge_centres = []
    wedge_frames  = []
    for i, rec in enumerate(records):
        cx, cy = rec.state.x, rec.state.y
        if wedge_centres:
            px, py = wedge_centres[-1]
            if math.hypot(cx - px, cy - py) < wedge_gap:
                continue
        wedge_centres.append((cx, cy))
        wedge_frames.append(i)

    # Resolve which encounter type to visualise throughout the panel.
    # force_enc takes priority; otherwise fall back to scenario_name.
    active_enc = force_enc if (force_enc and force_enc in _ACTIVE_SECTOR) \
                 else scenario_name

    _, sec_clr, rule_label_full, s_off, e_off = _ACTIVE_SECTOR[active_enc]

    for i, frame_idx in enumerate(wedge_frames):
        rec  = records[frame_idx]
        cx, cy, psi = rec.state.x, rec.state.y, rec.state.psi
        heading_deg = math.degrees(psi)
        theta1 = heading_deg + s_off
        theta2 = heading_deg + e_off

        # Filled wedge — very faint so stacking reads as continuous lane
        w_fill = Wedge((cx, cy), sector_radius, theta1, theta2,
                       facecolor=sec_clr, edgecolor="none",
                       alpha=0.13, lw=0, zorder=1)
        ax.add_patch(w_fill)

        # Edge outline — slightly more opaque to show sector boundary
        w_edge = Wedge((cx, cy), sector_radius, theta1, theta2,
                       facecolor="none", edgecolor=sec_clr,
                       alpha=0.35, lw=0.8, zorder=2)
        ax.add_patch(w_edge)

    # One prominent wedge at CPA to anchor attention
    cpa_idx = int(np.argmin([r.min_distance for r in records]))
    cpa_rec = records[cpa_idx]
    cx_c, cy_c, psi_c = cpa_rec.state.x, cpa_rec.state.y, cpa_rec.state.psi
    heading_c = math.degrees(psi_c)
    w_cpa = Wedge((cx_c, cy_c), sector_radius,
                  heading_c + s_off, heading_c + e_off,
                  facecolor=sec_clr, edgecolor=sec_clr,
                  alpha=0.38, lw=2.0, zorder=5)
    ax.add_patch(w_cpa)
    # Label inside the CPA wedge
    mid_a = math.radians((heading_c + s_off + heading_c + e_off) / 2.0)
    lx = cx_c + sector_radius * 0.52 * math.cos(mid_a)
    ly = cy_c + sector_radius * 0.52 * math.sin(mid_a)
    short = rule_label_full.split("·")[1].strip()
    ax.text(lx, ly, short, fontsize=14, ha="center", va="center",
            color=sec_clr, fontweight="bold", zorder=11,
            path_effects=[_pe.withStroke(linewidth=2.5, foreground="white")])

    # ── Pass 2: CC-CBF barrier at non-overlapping barrier intervals ───────
    placed_centres = []
    placed_frames  = []
    for i, rec in enumerate(records):
        cx, cy = rec.state.x, rec.state.y
        if placed_centres:
            px, py = placed_centres[-1]
            if math.hypot(cx - px, cy - py) < min_gap:
                continue
        placed_centres.append((cx, cy))
        placed_frames.append(i)

    dists = []
    for i in placed_frames:
        rec = records[i]
        dists.append(
            math.hypot(rec.obstacles[0].x - rec.state.x,
                       rec.obstacles[0].y - rec.state.y)
            if rec.obstacles else 999.0
        )

    for frame_idx, dist in zip(placed_frames, dists):
        rec = records[frame_idx]
        cx, cy, psi = rec.state.x, rec.state.y, rec.state.psi

        # Physics-accurate: use actual h_cc to scale the lobe inflation
        h_val     = _h_cc_at(rec, lam)
        lam_draw  = _effective_lam(h_val, lam)
        tightness = 1.0 - max(0.0, min(1.0, (h_val - _H_FULL) / (_H_SLACK - _H_FULL)))
        bx, by = cbf_contour(cx, cy, psi, lam_draw, theta_c)
        proximity  = math.exp(-dist / (1.8 * r_max))
        fill_alpha = 0.04 + 0.18 * tightness * (0.5 + 0.5 * proximity)
        edge_alpha = 0.20 + 0.80 * tightness * (0.5 + 0.5 * proximity)
        ax.fill(bx, by, color=accent, alpha=fill_alpha, zorder=6)
        ax.plot(bx, by, color=accent, alpha=edge_alpha, lw=1.6, zorder=7)

    # ── Safety circle at CPA obstacle ─────────────────────────────────────
    if cpa_rec.obstacles:
        ox, oy = cpa_rec.obstacles[0].x, cpa_rec.obstacles[0].y
        ax.add_patch(Circle((ox, oy), cfg.D_SAFE,
                            fill=False, ec=C_SAFE, lw=2.0,
                            ls=":", zorder=8, alpha=0.75))

    # ── Ship hulls at START position (ghost) ──────────────────────────────
    start_rec = records[0]
    hx_s, hy_s = hull_polygon(start_rec.state.x, start_rec.state.y,
                               start_rec.state.psi,
                               cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx_s, hy_s, color=C_OWN, zorder=7, alpha=0.30)
    ax.plot(hx_s, hy_s, color=C_OWN, lw=0.8,  zorder=7, alpha=0.55)

    if start_rec.obstacles:
        os_     = start_rec.obstacles[0]
        psi_os  = math.atan2(os_.vy, os_.vx) if (abs(os_.vx) + abs(os_.vy)) > 0.1 \
                  else 0.0
        hx_os, hy_os = hull_polygon(os_.x, os_.y, psi_os,
                                     cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx_os, hy_os, color=C_OBS, zorder=7, alpha=0.30)
        ax.plot(hx_os, hy_os, color=C_OBS, lw=0.8,  zorder=7, alpha=0.55)

    # ── Ship hulls at CPA (solid) ─────────────────────────────────────────
    hx, hy = hull_polygon(cpa_rec.state.x, cpa_rec.state.y,
                           cpa_rec.state.psi,
                           cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx, hy, color=C_OWN, zorder=9, alpha=0.95)
    ax.plot(hx, hy, color="white", lw=1.2, zorder=10)
    if cpa_rec.obstacles:
        o      = cpa_rec.obstacles[0]
        psi_ob = math.atan2(o.vy, o.vx) if (abs(o.vx) + abs(o.vy)) > 0.1 else 0.0
        hx2, hy2 = hull_polygon(o.x, o.y, psi_ob,
                                  cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx2, hy2, color=C_OBS, zorder=9, alpha=0.95)
        ax.plot(hx2, hy2, color="white", lw=1.2, zorder=10)

    # ── Ship hulls at end (ghost) ─────────────────────────────────────────
    end_rec = records[-1]
    hx_e, hy_e = hull_polygon(end_rec.state.x, end_rec.state.y,
                               end_rec.state.psi,
                               cfg.USV_LENGTH, cfg.USV_WIDTH)
    ax.fill(hx_e, hy_e, color=C_OWN, zorder=8, alpha=0.45)
    ax.plot(hx_e, hy_e, color=C_OWN, lw=0.8, zorder=8, alpha=0.70)
    if end_rec.obstacles:
        oe     = end_rec.obstacles[0]
        psi_oe = math.atan2(oe.vy, oe.vx) if (abs(oe.vx) + abs(oe.vy)) > 0.1 else 0.0
        hx_oe, hy_oe = hull_polygon(oe.x, oe.y, psi_oe,
                                     cfg.USV_LENGTH, cfg.USV_WIDTH)
        ax.fill(hx_oe, hy_oe, color=C_OBS, zorder=8, alpha=0.45)
        ax.plot(hx_oe, hy_oe, color=C_OBS, lw=0.8, zorder=8, alpha=0.70)

    # ── Direction arrows near CPA ─────────────────────────────────────────
    n    = len(records)
    mid  = max(0, cpa_idx - n // 10)
    step = max(1, n // 50)
    if mid + step < len(own_x):
        ax.annotate("", xy=(own_x[mid + step], own_y[mid + step]),
                    xytext=(own_x[mid], own_y[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OWN,
                                   lw=1.5, mutation_scale=18), zorder=12)
    if obs_x and mid + step < len(obs_x):
        ax.annotate("", xy=(obs_x[mid + step], obs_y[mid + step]),
                    xytext=(obs_x[mid], obs_y[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=C_OBS,
                                   lw=1.5, mutation_scale=18), zorder=12)

    # ── Rule badge (top-right) ────────────────────────────────────────────
    man_at_cpa = (cpa_rec.command.manoeuvre.replace("_", " ").title()
                  if cpa_rec.command else "Hold Course")
    ax.text(xlim[1] - xspan * 0.03,
            ylim[1] - yspan * 0.03,
            f"{rule_label_full}\n{man_at_cpa}",
            ha="right", va="top", fontsize=20, fontweight="bold",
            color=sec_clr, zorder=14,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=sec_clr, linewidth=2.0, alpha=0.92))

    # ── Scenario title (top-left) ─────────────────────────────────────────
    ax.text(xlim[0] + xspan * 0.03,
            ylim[1] - yspan * 0.03,
            title, ha="left", va="top",
            fontsize=17, fontweight="bold", color="#0D2D6B",
            zorder=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#0D2D6B", linewidth=2.0, alpha=0.92))


def add_legend_active_rule(fig):
    """Legend for the active-rule figure."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    handles = [
        mlines.Line2D([], [], color=C_OWN,  lw=2.2, label="USV path"),
        mlines.Line2D([], [], color=C_OBS,  lw=2.2, label="Obstacle path"),
        mlines.Line2D([], [], color=C_SAFE, lw=1.2, ls=":",
                      label="Safety circle"),
        mpatches.Patch(facecolor="#FB8C00", alpha=0.55,
                       label="CC-CBF barrier (Overtaking)"),
        mpatches.Patch(facecolor="#1565C0", alpha=0.55,
                       label="CC-CBF barrier (Head-On)"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.55,
                       label="Rule 14 · Head-On sector"),
        mpatches.Patch(facecolor="#42A5F5", alpha=0.55,
                       label="Rule 13 · Overtaking sector"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=20, frameon=True, framealpha=0.92,
               edgecolor="#AAAAAA", bbox_to_anchor=(0.5, 0.00))


def add_legend(fig):
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=C_OWN,  lw=2.2,
                      label="USV path"),
        mlines.Line2D([], [], color=C_OBS,  lw=2.2,
                      label="Obstacle path"),
        mlines.Line2D([], [], color=C_SAFE, lw=1.2, ls=":",
                      label="Safety circle"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=20, frameon=True, framealpha=0.92,
               edgecolor="#AAAAAA", bbox_to_anchor=(0.5, 0.00))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("Running simulations ...")
    sim_data = {}
    for sname, *_ in SCENARIOS:
        print(f"  {sname} ...", end=" ", flush=True)
        sc, records = simulate(sname)
        sim_data[sname] = (sc, records)
        print(f"done  ({len(records)} steps)")

    suptitle = "Collision Avoidance with COLREGS-Compliant CC-CBF"

    # ── Figure 1: stacked (top/bottom) ───────────────────────────────────
    print("Rendering stacked figure ...")
    out_stacked = OUT_PATH.replace(".png", "_stacked.png")
    fig1, axes1 = plt.subplots(
        2, 1, figsize=(18, 14),
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )
    fig1.patch.set_facecolor("white")
    for row, (sname, title, rule, accent, lam, tc) in enumerate(SCENARIOS):
        sc, records = sim_data[sname]
        extra_slots = {3, 6} if sname == "head_on" else None
        render_panel(fig1, axes1[row], sc, records, title, rule, accent, lam, tc,
                     show_obs_end=(sname != "head_on"),
                     extra_inflated_slots=extra_slots)
    add_legend(fig1)
    fig1.suptitle(suptitle, fontsize=20, fontweight="bold",
                  color="#0D2D6B", y=0.997)
    fig1.savefig(out_stacked, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    fig1.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    print(f"Saved → {out_stacked}  (also → {OUT_PATH})")

    # ── Figure 2: side by side (left/right) ──────────────────────────────
    print("Rendering side-by-side figure ...")
    out_sidebyside = OUT_PATH.replace(".png", "_sidebyside.png")
    fig2, axes2 = plt.subplots(
        1, 2, figsize=(22, 8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.06},
    )
    fig2.patch.set_facecolor("white")
    for col, (sname, title, rule, accent, lam, tc) in enumerate(SCENARIOS):
        sc, records = sim_data[sname]
        extra_slots = {3, 6} if sname == "head_on" else None
        render_panel(fig2, axes2[col], sc, records, title, rule, accent, lam, tc,
                     show_obs_end=(sname != "head_on"),
                     extra_inflated_slots=extra_slots)
    add_legend(fig2)
    fig2.suptitle(suptitle, fontsize=20, fontweight="bold",
                  color="#0D2D6B", y=0.997)
    fig2.savefig(out_sidebyside, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    print(f"Saved → {out_sidebyside}")

    # ── Figure 3: barrier-shape style — stacked ───────────────────────────
    # Barrier colours & transparency matching paper/figures/barrier_shape.png
    # (flat fill alpha=0.10, edge lw=1.8, Head-On=#1565C0, Overtaking=#FB8C00)
    print("Rendering barrier-shape-style stacked figure ...")
    out_bs_stacked = OUT_PATH.replace(".png", "_bshape_stacked.png")
    fig3, axes3 = plt.subplots(
        2, 1, figsize=(18, 14),
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )
    fig3.patch.set_facecolor("white")
    for row, (sname, title, rule, accent, lam, tc) in enumerate(SCENARIOS_BARRIER_STYLE):
        sc, records = sim_data[sname]
        extra_slots = {3, 6} if sname == "head_on" else None
        render_panel(fig3, axes3[row], sc, records, title, rule, accent, lam, tc,
                     bshape=True, show_obs_end=(sname != "head_on"),
                     extra_inflated_slots=extra_slots)
    add_legend(fig3)
    fig3.suptitle(suptitle, fontsize=20, fontweight="bold",
                  color="#0D2D6B", y=0.997)
    fig3.savefig(out_bs_stacked, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    print(f"Saved → {out_bs_stacked}")

    # ── Figure 4: barrier-shape style — side by side ──────────────────────
    print("Rendering barrier-shape-style side-by-side figure ...")
    out_bs_sidebyside = OUT_PATH.replace(".png", "_bshape_sidebyside.png")
    fig4, axes4 = plt.subplots(
        1, 2, figsize=(22, 8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.06},
    )
    fig4.patch.set_facecolor("white")
    for col, (sname, title, rule, accent, lam, tc) in enumerate(SCENARIOS_BARRIER_STYLE):
        sc, records = sim_data[sname]
        extra_slots = {3, 6} if sname == "head_on" else None
        render_panel(fig4, axes4[col], sc, records, title, rule, accent, lam, tc,
                     bshape=True, show_obs_end=(sname != "head_on"),
                     extra_inflated_slots=extra_slots)
    add_legend(fig4)
    fig4.suptitle(suptitle, fontsize=20, fontweight="bold",
                  color="#0D2D6B", y=0.997)
    fig4.savefig(out_bs_sidebyside, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    print(f"Saved → {out_bs_sidebyside}")

    # ── Figure 5: COLREG situations + CC-CBF barrier (stacked) ───────────
    # Each snapshot shows all four COLREG sector wedges around the USV
    # (active sector highlighted) so the reader sees how the barrier lobe
    # aligns with the encounter geometry — same insight as the animated GIFs.
    print("Rendering COLREG-situations + CC-CBF figure ...")
    out_colreg = OUT_PATH.replace(".png", "_colreg_situations.png")
    fig5, axes5 = plt.subplots(
        2, 1, figsize=(18, 14),
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )
    fig5.patch.set_facecolor("white")
    for row, (sname, title, rule, accent, lam, tc) in enumerate(SCENARIOS_BARRIER_STYLE):
        sc, records = sim_data[sname]
        render_panel_colreg(fig5, axes5[row], sc, records, sname,
                            title, rule, accent, lam, tc)
    add_legend_colreg(fig5)
    fig5.suptitle(
        "COLREGS Encounter Situations & CC-CBF Barrier Evolution",
        fontsize=20, fontweight="bold", color="#0D2D6B", y=0.997,
    )
    fig5.savefig(out_colreg, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    print(f"Saved → {out_colreg}")

    # ── Figure 6: active COLREG rule only + CC-CBF barrier ───────────────
    # Panel 0 (Overtaking) → force Rule 13 "overtaking" sector throughout
    # Panel 1 (Head-On)    → force Rule 14 "head_on"    sector throughout
    print("Rendering active-rule figure ...")
    out_active = OUT_PATH.replace(".png", "_active_rule.png")
    fig6, axes6 = plt.subplots(
        2, 1, figsize=(18, 14),
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )
    fig6.patch.set_facecolor("white")
    # Explicit per-row encounter type — one-to-one with SCENARIOS_BARRIER_STYLE
    _force_encs = ["overtaking", "head_on"]
    for row, (sname, title, rule, accent, lam, tc) in enumerate(SCENARIOS_BARRIER_STYLE):
        sc, records = sim_data[sname]
        render_panel_active_rule(fig6, axes6[row], sc, records, sname,
                                 title, rule, accent, lam, tc,
                                 force_enc=_force_encs[row])
    add_legend_active_rule(fig6)
    fig6.suptitle(
        "Active COLREGS Rule & CC-CBF Barrier Evolution",
        fontsize=20, fontweight="bold", color="#0D2D6B", y=0.997,
    )
    fig6.savefig(out_active, dpi=DPI, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    print(f"Saved → {out_active}")


if __name__ == "__main__":
    main()
