#!/usr/bin/env python3
"""
Fig. 5: representative trajectories of all five controllers in the four
canonical scenarios (seed 42, the runs of Table III).

Usage:
    python generate_trajectories.py [--out-dir figures]

Output: figures/traj.jpg
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
from matplotlib.lines import Line2D                 # noqa: E402

from algorithms.registry import CONTROLLERS        # noqa: E402
from core.entities import USVParameters             # noqa: E402
from core.evaluation import FIGURES_DIR, add_common_args, ensure_dir  # noqa: E402
from core.scenario import get_scenario              # noqa: E402
from core.simulator import Simulator                # noqa: E402

# (legend label, registry name, colour, line style, line width)
CTRL = [("CC-CBF (Ours)", "CC-CBF", "#1565c0", "-", 2.4),
        ("C3BF [4]", "C3BF", "#e53935", (0, (5, 2.5)), 1.8),
        ("Rule-COLREG [6]", "Rule-COLREG", "#43a047", (0, (6, 2, 1.5, 2)), 1.8),
        ("Geo-CRI [9]", "Geo-CRI", "#fb8c00", (0, (1.2, 1.8)), 2.0),
        ("TC-CBF [14]", "TC-CBF", "#7b1fa2", (0, (5, 1.5, 1.2, 1.5, 1.2, 1.5)), 1.8)]
SCN = [("head_on", "Head-On (Rule 14)"), ("crossing_give_way", "Crossing Give-Way (Rule 15)"),
       ("overtaking", "Overtaking (Rule 13)"), ("static_obstacles", "Static Obstacle Field")]
SEED = 42


def main():
    ap = argparse.ArgumentParser(description="Fig. 5: representative trajectories.")
    add_common_args(ap, workers=False, out_dir=FIGURES_DIR)
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.3))
    for ax, (scn, title) in zip(axes.ravel(), SCN):
        sc = get_scenario(scn, seed=SEED)
        first = None
        for label, name, col, ls, lw in CTRL:
            sim = Simulator(get_scenario(scn, seed=SEED), CONTROLLERS[name](), USVParameters())
            sim.run_to_completion()
            xs = [r.state.x for r in sim.records if r.state]
            ys = [r.state.y for r in sim.records if r.state]
            ax.plot(xs, ys, color=col, ls=ls, lw=lw, zorder=4 if name == "CC-CBF" else 3)
            if first is None:
                first = sim.records
        # targets (from the first run)
        for j, t in enumerate(sc.targets):
            tx = [r.obstacles[j].x for r in first if len(r.obstacles) > j]
            ty = [r.obstacles[j].y for r in first if len(r.obstacles) > j]
            if getattr(t, "is_static", False) or t.speed < 1e-3:
                L, W = t.length, t.width
                c, s = math.cos(t.psi), math.sin(t.psi)
                poly = np.array([[L / 2, 0], [0.2 * L, -W / 2], [-L / 2, -W / 2],
                                 [-L / 2, W / 2], [0.2 * L, W / 2]])
                poly = poly @ np.array([[c, s], [-s, c]]) + [t.x, t.y]
                ax.fill(poly[:, 0], poly[:, 1], color="0.55", alpha=0.8, zorder=2)
            else:
                ax.plot(tx, ty, color="#e53935", lw=1.2, alpha=0.55, zorder=2)
                ax.annotate("", xy=(tx[-1], ty[-1]), xytext=(tx[-8], ty[-8]),
                            arrowprops=dict(arrowstyle="-|>", color="#e53935", lw=1.2, alpha=0.8))
        s0 = sc.ownship_start
        ax.plot(s0.x, s0.y, "o", color="#6a1b9a", ms=5, zorder=6)
        ax.plot(sc.ownship_goal_x, sc.ownship_goal_y, "*", color="#2e7d32", ms=12, mec="k",
                mew=0.5, zorder=6)
        ax.set_title(title)
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=w) for _, _, c, ls, w in CTRL]
    fig.legend(handles, [label for label, *_ in CTRL], loc="lower center", ncol=3, frameon=True,
               fontsize=8.5, handlelength=3.0, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(ensure_dir(args.out_dir), "traj.jpg")
    fig.savefig(out, dpi=300, pil_kwargs={"quality": 95})
    plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
