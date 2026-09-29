"""Smoke tests: every experiment script runs end to end on a tiny budget."""
import csv
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script, *args, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, script), *args],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_run_all_planners(tmp_path):
    out = run("run_all_planners.py", "--scenario", "head_on", "--controller", "CC-CBF",
              "--out-dir", str(tmp_path), tmp_path=tmp_path)
    assert "CC-CBF" in out
    (r,) = rows(tmp_path / "table3_seed42.csv")
    assert round(float(r["min_separation"]), 2) == 19.15


def test_run_mc_5way(tmp_path):
    run("run_mc_5way.py", "--trials", "2", "--controllers", "CC-CBF", "C3BF",
        "--scenarios", "head_on", "--workers", "2", "--out-dir", str(tmp_path), tmp_path=tmp_path)
    rs = rows(tmp_path / "results_mc_5way.csv")
    assert len(rs) == 4 and {"reached_goal", "runtime"} <= set(rs[0])


def test_run_ablation(tmp_path):
    out = run("run_ablation.py", "--trials", "2", "--variants", "full", "no_nominal_shaping",
              "c3bf", "--scenarios", "head_on", "--workers", "1", "--out-dir", str(tmp_path),
              tmp_path=tmp_path)
    assert "McNemar" in out
    assert len(rows(tmp_path / "ablation_trials.csv")) == 6
    assert len(rows(tmp_path / "ablation_summary.csv")) == 3


def test_run_ablation_rejects_unknown_variant(tmp_path):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "run_ablation.py"),
                           "--variants", "bogus", "--out-dir", str(tmp_path)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "unknown ablation variant" in proc.stderr


def test_run_sensitivity(tmp_path):
    run("run_sensitivity.py", "--trials", "1", "--workers", "1", "--out-dir", str(tmp_path),
        tmp_path=tmp_path)
    assert len(rows(tmp_path / "results_lambda_sweep.csv")) == 15
    assert len(rows(tmp_path / "results_noise_robustness.csv")) == 15


def test_run_multi_vessel(tmp_path):
    out = run("run_multi_vessel.py", "--trials", "1", "--workers", "1", "--out-dir",
              str(tmp_path), tmp_path=tmp_path)
    assert "parallel_head_on" in out and "mixed_rules" in out


def test_run_solver_stats(tmp_path):
    out = run("run_solver_stats.py", "--mc-trials", "1", "--ablation-trials", "1",
              "--workers", "1", tmp_path=tmp_path)
    assert "ALL" in out


def test_generate_barrier_shape(tmp_path):
    run("generate_barrier_shape.py", "--out-dir", str(tmp_path), tmp_path=tmp_path)
    assert (tmp_path / "barrier_shape.png").exists()


def test_generate_hcc_traces(tmp_path):
    run("generate_hcc_traces.py", "--fig-dir", str(tmp_path), "--out-dir", str(tmp_path),
        tmp_path=tmp_path)
    assert (tmp_path / "hcc_traces.png").exists()
    assert (tmp_path / "results_hcc_traces.csv").exists()


def test_generate_multi_obstacle(tmp_path):
    run("generate_multi_obstacle.py", "--no-gif", "--out-dir", str(tmp_path), tmp_path=tmp_path)
    assert (tmp_path / "multi_obstacle_parallel_head_on.png").exists()
    assert (tmp_path / "multi_obstacle_mixed_rules.png").exists()


@pytest.mark.slow
def test_generate_trajectories(tmp_path):
    run("generate_trajectories.py", "--out-dir", str(tmp_path), tmp_path=tmp_path)
    assert (tmp_path / "traj.jpg").exists()
