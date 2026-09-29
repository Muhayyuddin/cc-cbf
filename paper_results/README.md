# Archived results behind the paper

These are the raw outputs from which every number in the manuscript was taken. They are kept as they were produced; the `.txt` files are console logs, so they show the absolute paths of the machine that generated them. To regenerate a result, run the command in the last column from the repository root. New outputs are written to `results/` and `figures/`, never to this folder.

| Paper item | Archived file(s) | Regenerate with |
|---|---|---|
| Table III: deterministic comparison, seed 42 | `table3_final.txt` | `python run_all_planners.py` |
| Table III: Time column (µs per update) | `controller_time.txt` | `python run_all_planners.py --repeats 3` |
| Table IV: component ablation (N = 200 paired trials) | `ablation_final.txt`, `ablation_final_trials.csv` | `python run_ablation.py` |
| Table V: λ sweep | `sensitivity_final.txt`, `results_lambda_sweep_final.csv` | `python run_sensitivity.py --tables A` |
| Table VI: heading-noise robustness | `sensitivity_final.txt`, `results_noise_robustness_final.csv` | `python run_sensitivity.py --tables B` |
| Table VII: Monte Carlo (1000 runs) | `mc_final.txt`, `results_mc_5way_final.csv` | `python run_mc_5way.py` |
| Sec. VI-H: solver-branch statistics | `relax_stats.txt` | `python run_solver_stats.py` |
| Sec. VI-G: multi-vessel statistics | `multi_final.txt` | `python run_multi_vessel.py` |
| Fig. 3: barrier shapes | `barrier_final.txt` | `python generate_barrier_shape.py` |
| Fig. 5: trajectories | `fig5.txt` | `python generate_trajectories.py` |
| Fig. 8: h_CC traces | `hcc_final.txt`, `results_hcc_traces_final.csv` | `python generate_hcc_traces.py` |
| Fig. 9: multi-vessel scenarios | `fig9.txt` | `python generate_multi_obstacle.py --no-gif` |

`ablation_final_trials.csv` has one row per trial (`seed` = 300 + trial index). The ablation log also covers extra lobe variants that the text mentions but the table does not show (`mirrored`, `stbd_lobes`, `ot_flip` and their `no_nominal_shaping_*` forms). To run them all, use `python run_ablation.py --variants all`.

## Reproducibility

All scripts are deterministic: seeds are fixed and results come back in job order whatever the number of workers. Re-running the scripts gave the following against the archived files:

* Tables III, V and VI, the multi-vessel and solver statistics, and the Fig. 8 traces are identical.
* Table VII is identical at the paper's reported precision. On a different machine, 5 of the 1000 per-trial rows differed in the third decimal. All five are TC-CBF overtaking trials, and the cause is floating-point rounding between platforms, which a few sensitive trajectories amplify.
* Table IV: all 7800 trials have identical side, collision and commitment outcomes. The clearances differ by less than 1e-7 m.

The per-update computation times depend on the hardware.
