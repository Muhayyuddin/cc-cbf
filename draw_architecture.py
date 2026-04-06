"""
draw_architecture.py  –  First draft
Three-Layer COLREG Control Architecture block diagram.
Hand-laid-out in absolute inches.

Output: paper/figures/architecture_three_layer.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 12.0, 10.0   # inches
DPI = 300

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.invert_yaxis()
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, radius=0.12):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=lw, zorder=2)
    ax.add_patch(p)

def badge(ax, x, y, w, h, facecolor, label, textcolor="white"):
    p = mpatches.Rectangle((x, y), w, h,
                            facecolor=facecolor, edgecolor="none", zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=6,
            color=textcolor, fontweight="bold", zorder=4)

def label(ax, x, y, text, size=9, color="black",
          ha="center", va="center", bold=False):
    ax.text(x, y, text, fontsize=size, color=color,
            ha=ha, va=va,
            fontweight="bold" if bold else "normal", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.5):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->, head_width=0.20, head_length=0.14",
                    color=color, lw=lw),
                zorder=3)

def hline(ax, x0, x1, y, color="#333333", lw=1.5):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=3)

def vline(ax, x, y0, y1, color="#333333", lw=1.5):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=3)

# ── Layout constants ──────────────────────────────────────────────────────────
CX = FIG_W / 2   # horizontal centre

ROW_SENSOR  = 0.20
ROW_CLASS   = 1.25
ROW_LAYERS  = 2.55
ROW_QP      = 4.40
ROW_PD      = 5.85
ROW_USV     = 7.10

BOX_H_SMALL = 0.85
BOX_H_LAYER = 1.45
BOX_H_QP    = 1.10

W_TOP   = 3.80
W_LAYER = 3.00
W_QP    = 10.00
W_PD    = 4.00
W_USV   = 3.00

L1_X = 0.50
L2_X = (FIG_W - W_LAYER) / 2
L3_X = FIG_W - 0.50 - W_LAYER

L1_CX = L1_X + W_LAYER / 2
L2_CX = L2_X + W_LAYER / 2
L3_CX = L3_X + W_LAYER / 2

QP_X  = (FIG_W - W_QP) / 2
PD_X  = (FIG_W - W_PD) / 2
USV_X = (FIG_W - W_USV) / 2

SOFT_C = "#888580"

# ── 1. Sensor Input ───────────────────────────────────────────────────────────
SEN_X = CX - W_TOP / 2
box(ax, SEN_X, ROW_SENSOR, W_TOP, BOX_H_SMALL,
    facecolor="#F0EFE9", edgecolor="#5F5D58", lw=0.8)
label(ax, CX, ROW_SENSOR + 0.28, "Sensor Input", size=9.5, bold=True, color="#1A1A18")
label(ax, CX, ROW_SENSOR + 0.58, "AIS / RADAR / GPS", size=8, color="#5F5D58")

# ── 2. Encounter Classifier ───────────────────────────────────────────────────
CLS_X = CX - W_TOP / 2
box(ax, CLS_X, ROW_CLASS, W_TOP, BOX_H_SMALL,
    facecolor="#F0EFE9", edgecolor="#5F5D58", lw=0.8)
label(ax, CX, ROW_CLASS + 0.28, "Encounter Classifier", size=9.5, bold=True, color="#1A1A18")
label(ax, CX, ROW_CLASS + 0.58, "Geometric encounter type  τ", size=8, color="#5F5D58")

# Sensor → Classifier
arrow(ax, CX, ROW_SENSOR + BOX_H_SMALL, CX, ROW_CLASS)

# ── 3. τ dispatch fan  (Classifier → three layers) ───────────────────────────
FAN_START = ROW_CLASS + BOX_H_SMALL
FAN_MID   = ROW_LAYERS - 0.30

vline(ax, CX, FAN_START, FAN_MID)
hline(ax, L1_CX, L3_CX, FAN_MID)
arrow(ax, L1_CX, FAN_MID, L1_CX, ROW_LAYERS)
arrow(ax, L2_CX, FAN_MID, L2_CX, ROW_LAYERS)
arrow(ax, L3_CX, FAN_MID, L3_CX, ROW_LAYERS)

for cx_i in (L1_CX, L2_CX, L3_CX):
    label(ax, cx_i - 0.15, FAN_MID + 0.18, "τ",
          size=8, color="#7A6200", ha="right")

# ── 4a. Layer 1 – Barrier Asymmetry (HARD, purple) ───────────────────────────
box(ax, L1_X, ROW_LAYERS, W_LAYER, BOX_H_LAYER,
    facecolor="#EEEAFF", edgecolor="#534AB7", lw=2.0)
badge(ax, L1_X + 0.08, ROW_LAYERS + 0.08, 0.72, 0.30,
      facecolor="#3C348A", label="HARD", textcolor="#FFAAAA")
label(ax, L1_CX, ROW_LAYERS + 0.62, "Barrier Asymmetry", size=9.5, bold=True, color="#2E2860")
label(ax, L1_CX, ROW_LAYERS + 0.98, r"$R_{\rm CC}(\theta,\tau)$ — hard CBF",
      size=8, color="#4A5010")

# ── 4b. Layer 2 – Heading Bias (SOFT, green) ─────────────────────────────────
box(ax, L2_X, ROW_LAYERS, W_LAYER, BOX_H_LAYER,
    facecolor="#E1F5EC", edgecolor="#0F6E56", lw=0.8)
badge(ax, L2_X + 0.08, ROW_LAYERS + 0.08, 0.72, 0.30,
      facecolor="#065040", label="SOFT", textcolor="#A0FFD0")
label(ax, L2_CX, ROW_LAYERS + 0.62, "Heading Bias", size=9.5, bold=True, color="#074D10")
label(ax, L2_CX, ROW_LAYERS + 0.98, "Soft nominal nudge", size=8, color="#0E6B16")

# ── 4c. Layer 3 – Stern Waypoint (SOFT, orange) ──────────────────────────────
box(ax, L3_X, ROW_LAYERS, W_LAYER, BOX_H_LAYER,
    facecolor="#FEF0DC", edgecolor="#854B0B", lw=0.8)
badge(ax, L3_X + 0.08, ROW_LAYERS + 0.08, 0.72, 0.30,
      facecolor="#62340A", label="SOFT", textcolor="#FFD090")
label(ax, L3_CX, ROW_LAYERS + 0.62, "Stern Waypoint", size=9.5, bold=True, color="#621D05")
label(ax, L3_CX, ROW_LAYERS + 0.98, "Crossing GW only", size=8, color="#854B0B")

# ── 5. Arrows: layers → QP solver ────────────────────────────────────────────
LAYER_BOT = ROW_LAYERS + BOX_H_LAYER

# Layer 1 hard arrow (bold purple)
arrow(ax, L1_CX, LAYER_BOT, L1_CX, ROW_QP, color="#534AB7", lw=2.5)
label(ax, L1_CX - 0.18, LAYER_BOT + 0.35,
      r"CBF: $a^\top v \!\geq\! c$", size=7, color="#4A5010", ha="right")

# Layers 2 & 3 soft arrows (grey)
arrow(ax, L2_CX, LAYER_BOT, L2_CX, ROW_QP, color=SOFT_C, lw=1.2)
arrow(ax, L3_CX, LAYER_BOT, L3_CX, ROW_QP, color=SOFT_C, lw=1.2)
label(ax, L2_CX + 0.12, LAYER_BOT + 0.35,
      r"$u_{\rm nom}$ (×2)", size=7.5, color="#7A6200", ha="left")

# ── 6. QP Solver ─────────────────────────────────────────────────────────────
box(ax, QP_X, ROW_QP, W_QP, BOX_H_QP,
    facecolor="#E6F2FF", edgecolor="#1860A5", lw=1.2)
label(ax, CX, ROW_QP + 0.32, "CC-CBF QP Solver",
      size=10, bold=True, color="#0B3D6B")
label(ax, CX, ROW_QP + 0.68,
      "20 Hz — minimal deviation from nominal velocity",
      size=8, color="#1768DD")

# ── 7. QP → PD ───────────────────────────────────────────────────────────────
QP_BOT = ROW_QP + BOX_H_QP
arrow(ax, CX, QP_BOT, CX, ROW_PD)
label(ax, CX + 0.15, QP_BOT + 0.30, r"$v^*_o$",
      size=9, color="#2B3A6E", ha="left")

# ── 8. PD Autopilot ───────────────────────────────────────────────────────────
box(ax, PD_X, ROW_PD, W_PD, BOX_H_SMALL,
    facecolor="#F0EFE9", edgecolor="#5F5D58", lw=0.8)
label(ax, CX, ROW_PD + 0.28, "Low-level PD Autopilot",
      size=9.5, bold=True, color="#1A1A18")
label(ax, CX, ROW_PD + 0.58, "Heading and speed tracking",
      size=8, color="#5F5D58")

# ── 9. PD → USV ──────────────────────────────────────────────────────────────
PD_BOT = ROW_PD + BOX_H_SMALL
arrow(ax, CX, PD_BOT, CX, ROW_USV)

# ── 10. USV Dynamics ─────────────────────────────────────────────────────────
box(ax, USV_X, ROW_USV, W_USV, BOX_H_SMALL,
    facecolor="#F0EFE9", edgecolor="#5F5D58", lw=0.8)
label(ax, CX, ROW_USV + 0.28, "USV Dynamics",
      size=9.5, bold=True, color="#1A1A18")
label(ax, CX, ROW_USV + 0.58, "Marine vessel plant model",
      size=8, color="#5F5D58")

# ── 11. Legend ────────────────────────────────────────────────────────────────
LEG_X = 0.30
LEG_Y = ROW_PD + 0.10
label(ax, LEG_X, LEG_Y, "Legend", size=8.5, bold=True,
      color="#2B3A6E", ha="left")
ax.plot([LEG_X, LEG_X + 0.80], [LEG_Y + 0.38, LEG_Y + 0.38],
        color="#534AB7", lw=2.5)
label(ax, LEG_X + 0.90, LEG_Y + 0.38,
      "Hard guarantee (Theorem 2)", size=7.5, color="#333333", ha="left")
ax.plot([LEG_X, LEG_X + 0.80], [LEG_Y + 0.68, LEG_Y + 0.68],
        color=SOFT_C, lw=1.2)
label(ax, LEG_X + 0.90, LEG_Y + 0.68,
      "Soft aid (empirically confirmed)", size=7.5, color="#333333", ha="left")

# ── 12. Footer note ───────────────────────────────────────────────────────────
label(ax, CX, ROW_USV + BOX_H_SMALL + 0.28,
      "* Pairwise layer interactions formally proved — Proposition 3 (Section VI)",
      size=6.5, color="#7A6200")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("paper/figures", exist_ok=True)
out = "paper/figures/architecture_three_layer.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
