"""
Helpers shared by the experiment scripts: CPA-side scoring, statistics,
parallel execution and output locations.
"""

import math
import os
import statistics
from multiprocessing import Pool
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from core.entities import StepRecord
from core.geometry import wrap_angle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")   # generated tables (CSV)
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")   # generated figures / GIFs


def ensure_dir(path: str) -> str:
    """Create *path* if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# CPA-side scoring (paper Sec. VI-B, "selected CPA-side outcome")
# ---------------------------------------------------------------------------

def cpa_record(records: Sequence[StepRecord]) -> StepRecord:
    """Step of minimum hull-to-hull clearance."""
    return min(records, key=lambda r: r.min_distance)


def correct_side(rec: StepRecord, scenario: str) -> bool:
    """
    Passing side of the (first) target at step *rec*:

    * head-on / overtaking: target on the own ship's port side
      (port-to-port passage; overtaken vessel passed on its starboard side);
    * crossing give-way: own ship astern of the target (behind its beam line).
    """
    if rec.state is None or not rec.obstacles:
        return True
    own, obs = rec.state, rec.obstacles[0]
    dx, dy = obs.x - own.x, obs.y - own.y
    if scenario in ("head_on", "overtaking"):
        return wrap_angle(math.atan2(dy, dx) - own.psi) > 0.0
    return ((own.x - obs.x) * math.cos(obs.psi)
            + (own.y - obs.y) * math.sin(obs.psi)) < 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample standard deviation (0 for fewer than two values)."""
    if not values:
        return float("nan"), float("nan")
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) >= 2 else 0.0)


def fmt_mean_sd(values: Sequence[float], digits: int = 2) -> str:
    m, s = mean_sd(values)
    return "N/A" if math.isnan(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def wilson_ci(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson 95 % score interval for k successes in n trials, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value of the 2x2 table [[a, b], [c, d]]."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def prob(x: int) -> float:
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    p_obs = prob(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(p for p in (prob(x) for x in range(lo, hi + 1))
                        if p <= p_obs * (1 + 1e-7)))


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def default_workers() -> int:
    """Worker processes to use by default (all cores but one)."""
    return max(1, (os.cpu_count() or 2) - 1)


def parallel_map(fn: Callable, jobs: Iterable, workers: Optional[int] = None,
                 chunksize: int = 1) -> List:
    """``list(map(fn, jobs))``, in *workers* processes when workers > 1.

    Results are returned in job order, so the output does not depend on the
    number of workers.  *fn* must be a module-level function.
    """
    jobs = list(jobs)
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with Pool(min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs, chunksize=chunksize)


def add_common_args(parser, workers: bool = True, out_dir: Optional[str] = RESULTS_DIR):
    """Add the ``--workers`` / ``--out-dir`` options used by every script."""
    if workers:
        parser.add_argument("--workers", type=int, default=None,
                            help="parallel worker processes (default: CPU count - 1; "
                                 "1 = serial)")
    if out_dir is not None:
        parser.add_argument("--out-dir", default=out_dir,
                            help="output directory (default: <repo>/%s)"
                                 % os.path.basename(out_dir))
    return parser
