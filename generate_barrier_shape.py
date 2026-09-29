#!/usr/bin/env python3
"""
Fig. 3: CC-CBF barrier shapes R_CC(theta) for the three primary encounters.

Only the directional lobe is shown (the Rule-15 stern-passage factor is
defined in the target frame).  lambda, theta_C and R_b are read from
algorithms.cc_cbf, so the figure always matches Table II.

Usage:
    python generate_barrier_shape.py [--out-dir figures]

Output: figures/barrier_shape.png / .pdf
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

import algorithms.cc_cbf as m                       # noqa: E402
from algorithms.cc_cbf import _phi_and_dphi         # noqa: E402
from core.evaluation import FIGURES_DIR, add_common_args, ensure_dir  # noqa: E402

PANELS = [("overtaking", "Overtaking\n(Rule 13)", "#f39c12"),
          ("head_on", "Head-On\n(Rule 14)", "#1f5fd1"),
          ("crossing_give_way", "Crossing GW\n(Rule 15)", "#e03a3a")]


def main():
    ap = argparse.ArgumentParser(description="Fig. 3: CC-CBF barrier shapes.")
    add_common_args(ap, workers=False, out_dir=FIGURES_DIR)
    args = ap.parse_args()
    out = os.path.join(ensure_dir(args.out_dir), "barrier_shape")

    th = np.linspace(-math.pi, math.pi, 721)
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection="polar"), figsize=(9.6, 4.4))
    for ax, (enc, title, col) in zip(axes, PANELS):
        r = np.array([m.R_BASE * (1 + m.LAMBDA[enc] * _phi_and_dphi(t, enc)[0]) for t in th])
        ax.plot(th, np.full_like(th, m.R_BASE), "--", color="#555", lw=1.4, label=r"$R_b$ (isotropic)")
        ax.plot(th, r, color=col, lw=2.6, label=r"$R_{CC}$ " + title.split("\n")[0])
        ax.fill(th, r, color=col, alpha=0.12)
        tc = m.THETA_C[enc]
        ax.plot([tc, tc], [0, 25.5], ":", color="k", lw=1.3)
        ax.text(tc, 27.5, r"$\theta_C$", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_rmax(26)
        ax.set_rticks([13, 19, 25])
        ax.set_rlabel_position(190)
        ax.tick_params(labelsize=10)
        ax.grid(True, color="#bbb")
    h = ([plt.Line2D([], [], ls="--", color="#555", lw=1.4)]
         + [plt.Line2D([], [], color=c, lw=2.6) for _, _, c in PANELS])
    fig.legend(h, [r"$R_b$ (isotropic)", r"$R_{CC}$ Overtaking", r"$R_{CC}$ Head-On",
                   r"$R_{CC}$ Crossing"],
               loc="lower center", ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print("theta_C (deg):", {k: round(math.degrees(v)) for k, v in m.THETA_C.items()},
          "->", out + ".png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
