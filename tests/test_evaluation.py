import math

import pytest

from core.evaluation import fisher_exact, mcnemar_exact, parallel_map, wilson_ci


def test_wilson_matches_paper_interval():
    # Full CC-CBF, head-on ablation: 173 / 200 side-correct -> [81.1, 90.6] %
    lo, hi = wilson_ci(173, 200)
    assert (round(lo, 1), round(hi, 1)) == (81.1, 90.6)


def test_mcnemar_exact_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    for b, c in [(0, 0), (3, 0), (10, 2), (7, 7), (40, 6)]:
        expected = 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue
        assert math.isclose(mcnemar_exact(b, c), expected, rel_tol=1e-9)


def test_fisher_exact_against_scipy_and_paper():
    stats = pytest.importorskip("scipy.stats")
    for table in [(0, 200, 14, 186), (0, 200, 15, 185), (3, 7, 5, 2), (1, 9, 11, 3)]:
        a, b, c, d = table
        assert math.isclose(fisher_exact(*table), stats.fisher_exact([[a, b], [c, d]])[1],
                            rel_tol=1e-6)
    # Paper Sec. VI-H: p ~ 1e-4 vs Rule-COLREG, ~ 5e-5 vs Geo-CRI
    assert f"{fisher_exact(0, 200, 14, 186):.0e}" == "1e-04"
    assert f"{fisher_exact(0, 200, 15, 185):.0e}" == "5e-05"


def _square(x):
    return x * x


def test_parallel_map_preserves_order():
    jobs = list(range(20))
    assert parallel_map(_square, jobs, workers=1) == [j * j for j in jobs]
    assert parallel_map(_square, jobs, workers=2) == [j * j for j in jobs]
