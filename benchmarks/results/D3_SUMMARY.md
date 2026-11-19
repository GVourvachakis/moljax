# D3: adaptive exponential work precision

## Setup and protocol

The H1 threshold and matched accuracy targets were fixed before comparison. The reference arrays and their checksums are in `d3_references.npz` and `d3_references.json`. CPU timings are secondary evidence.

| Problem | Model | Fields | Domain length | Final time | Diffusivities | Grids | RHS reads time | Regime |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| P1 | manufactured | u | 1 | 2 | u=0.01 | 64 | True | — |
| P2 | Gray–Scott | u, v | 1 | 100 | u=2e-05, v=1e-05 | 128, 256 | False | — |
| P3 | Turing Brusselator | u, v | 5 | 200 | u=0.01, v=0.1 | 64 | False | TURING_REGIME |
| P4 | Hopf Brusselator | u, v | 5 | 20 | u=0.01, v=0.02 | 64 | False | HOPF_REGIME |
| P5 | Schnakenberg | u, v | 1 | 1 | u=1, v=40 | 64, 128 | False | — |

Source: `benchmarks/d3_work_precision.py`, `PROBLEM_SETTINGS`, `CASES`, `setup()`; periodic FD `DiffusionOperator` on every case.

| Configuration | Meaning | NFE per attempt |
| --- | --- | ---: |
| S0 | ETDRK4 step doubling with the default controller | 12 (`/settings/NFE_per_attempt/S0`) |
| sensitivity | S0 with atol = rtol | 12 (`/settings/NFE_per_attempt/sensitivity`) |
| E0a | ERK32ZB embedded pair | 4 (`/settings/NFE_per_attempt/E0a`) |
| E0b | ERK43ZB embedded pair | 5 (`/settings/NFE_per_attempt/E0b`) |
| F | Uniform fixed ETDRK4 | 4 per fixed step (`benchmarks/d3_work_precision.py:fixed_cell`) |
| F_g | Graded fixed ETDRK4 on P5 | 4 per fixed step (`benchmarks/d3_work_precision.py:fixed_cell`) |

Source: `d3_work_precision.json`, `/settings/NFE_per_attempt`; `benchmarks/d3_work_precision.py`, `fixed_cell`.

The error metric is max fieldwise relative infinity error (`/settings/error_metric`). The validity rule is error_inf >= 100*est_err -> valid; 10*est_err <= error_inf < 100*est_err -> valid, reference-limited; error_inf < 10*est_err -> indeterminate (`/validity_rule`): it compares absolute error with the reference's `est_err` and selects records for the Pareto fronts. Q is S0 NFE divided by pair NFE at matched error; R uses F or F_g NFE as the numerator. Both use log–log interpolation on valid Pareto fronts without extrapolation. Lean NFE counts one reaction evaluation per step position: S0 uses `10*attempts + accepted`, E0a `3*attempts + 1`, and E0b `4*attempts + accepted` (`benchmarks/d3_work_precision.py:adaptive_cell`).

A target is confirmed when it is at least ten times its reference-confirmation level (`reference_confirmation/*/epsilon_conf`). Confirmation labels come from the [reference audit on the study branch](https://github.com/GVourvachakis/moljax/tree/f3a05c93a07a2cf3684660f94f410b989bd119e6). H1 uses every attainable valid target regardless of confirmation (`benchmarks/d3_work_precision_analysis.py:hypotheses`).

H1 requires Q ≥ 1.15 (`/settings/H1_thresholds/support_Q`) on at least 3 (`/settings/H1_thresholds/minimum_problem_count`) eligible problems, with no problem below 0.9 (`/settings/H1_thresholds/loss_Q`).

Frozen targets: 1e-3, 1e-4, 1e-5, 1e-6 (`/settings/targets`).

| Case | Reference confirmation level | Method |
| --- | ---: | --- |
| P1_64 | 0 (`/reference_confirmation/P1_64/epsilon_conf`) | exact |
| P2_128 | 1.586e-09 (`/reference_confirmation/P2_128/epsilon_conf`) | self-convergence in D3 metric |
| P2_256 | 1.719e-09 (`/reference_confirmation/P2_256/epsilon_conf`) | self-convergence in D3 metric |
| P3_64 | 5.524e-07 (`/reference_confirmation/P3_64/epsilon_conf`) | endpoint distance plus bounded self-error |
| P4_64 | 2.877e-08 (`/reference_confirmation/P4_64/epsilon_conf`) | endpoint distance plus bounded self-error |
| P5_64 | 9.065e-06 (`/reference_confirmation/P5_64/epsilon_conf`) | endpoint distance plus bounded self-error |
| P5_128 | 5.549e-07 (`/reference_confirmation/P5_128/epsilon_conf`) | endpoint distance plus bounded self-error |

Source: `d3_work_precision.json`, `/settings`, `/reference_confirmation`, `/validity_rule`.

All 357 records were assembled (`/records`); every adaptive run succeeded. The 15 unsuccessful records are coarse fixed-step rungs that became non-finite (`/stage_a/failures`). Exclusions by case and configuration:

| Case / configuration | Excluded records |
| --- | ---: |
| P2/128/F | 2 |
| P2/128/sensitivity | 2 |
| P2/256/F | 2 |
| P2/256/sensitivity | 3 |
| P3/64/E0a | 5 |
| P3/64/E0b | 4 |
| P3/64/F | 1 |
| P3/64/S0 | 6 |
| P3/64/sensitivity | 6 |
| P4/64/E0a | 4 |
| P4/64/E0b | 3 |
| P4/64/S0 | 3 |
| P4/64/sensitivity | 4 |
| P5/64/E0a | 1 |
| P5/64/F | 1 |
| P5/64/F_g | 1 |
| P5/64/sensitivity | 1 |
| P5/128/E0a | 2 |
| P5/128/E0b | 2 |
| P5/128/F | 2 |
| P5/128/F_g | 1 |
| P5/128/S0 | 1 |
| P5/128/sensitivity | 1 |

Source: `d3_work_precision.json`, `/stage_a/excluded_counts`.

## H1: embedded pairs against step doubling

| Pair | Problem | Median Q | ≥ 1.15 | < 0.9 |
| --- | --- | ---: | --- | --- |
| E0a | P1 | 1.151 (`/hypotheses/H1/E0a/by_problem/P1/median_Q`) | True | False |
| E0a | P2 | 0.7416 (`/hypotheses/H1/E0a/by_problem/P2/median_Q`) | False | True |
| E0a | P3 | 0.7304 (`/hypotheses/H1/E0a/by_problem/P3/median_Q`) | False | True |
| E0a | P4 | 0.386 (`/hypotheses/H1/E0a/by_problem/P4/median_Q`) | False | True |
| E0a | P5 | 1.351 (`/hypotheses/H1/E0a/by_problem/P5/median_Q`) | True | False |
| E0b | P2 | 1.865 (`/hypotheses/H1/E0b/by_problem/P2/median_Q`) | True | False |
| E0b | P3 | 2.377 (`/hypotheses/H1/E0b/by_problem/P3/median_Q`) | True | False |
| E0b | P4 | 1.206 (`/hypotheses/H1/E0b/by_problem/P4/median_Q`) | True | False |
| E0b | P5 | 3.349 (`/hypotheses/H1/E0b/by_problem/P5/median_Q`) | True | False |

Source: `d3_work_precision.json`, `/hypotheses/H1/*/by_problem/*/median_Q`.

E0a: 2 (`/hypotheses/H1/E0a/positive_problem_count`) of 5 (`/hypotheses/H1/E0a/eligible_problem_count`) eligible problems meet the positive threshold; any below the loss threshold: True (`/hypotheses/H1/E0a/any_problem_below_0p90`); supported: False (`/hypotheses/H1/E0a/supported`).
E0b: 4 (`/hypotheses/H1/E0b/positive_problem_count`) of 4 (`/hypotheses/H1/E0b/eligible_problem_count`) eligible problems meet the positive threshold; any below the loss threshold: False (`/hypotheses/H1/E0b/any_problem_below_0p90`); supported: True (`/hypotheses/H1/E0b/supported`).
ERK43ZB has no P1 row because its time-dependent RHS is incompatible with the pair's one-sixth stage under the tick-clock rounding guarantee.

The primary Q uses nominal NFE. Lean NFE and CPU-time ratios are reported as secondary checks.

| JSON key | Target | Q | Lean Q | Confirmed |
| --- | ---: | ---: | ---: | --- |
| P1/64/E0a | 1e-3 | unattainable | unattainable | True |
| P1/64/E0a | 1e-4 | 1.421 | 1.709 | True |
| P1/64/E0a | 1e-5 | 1.151 | 1.393 | True |
| P1/64/E0a | 1e-6 | 0.8447 | 1.027 | True |
| P2/128/E0a | 1e-3 | unattainable | unattainable | True |
| P2/128/E0a | 1e-4 | 0.9601 | 1.16 | True |
| P2/128/E0a | 1e-5 | 0.7428 | 0.9011 | True |
| P2/128/E0a | 1e-6 | 0.5587 | 0.6812 | True |
| P2/128/E0b | 1e-3 | unattainable | unattainable | True |
| P2/128/E0b | 1e-4 | 1.911 | 1.762 | True |
| P2/128/E0b | 1e-5 | 1.846 | 1.701 | True |
| P2/128/E0b | 1e-6 | 1.877 | 1.725 | True |
| P2/256/E0a | 1e-3 | unattainable | unattainable | True |
| P2/256/E0a | 1e-4 | 0.961 | 1.161 | True |
| P2/256/E0a | 1e-5 | 0.7404 | 0.8986 | True |
| P2/256/E0a | 1e-6 | 0.549 | 0.6694 | True |
| P2/256/E0b | 1e-3 | unattainable | unattainable | True |
| P2/256/E0b | 1e-4 | 1.936 | 1.776 | True |
| P2/256/E0b | 1e-5 | 1.852 | 1.706 | True |
| P2/256/E0b | 1e-6 | 1.853 | 1.704 | True |
| P3/64/E0a | 1e-3 | 0.7304 | 0.8926 | True |
| P3/64/E0a | 1e-4 | unattainable | unattainable | True |
| P3/64/E0a | 1e-5 | unattainable | unattainable | True |
| P3/64/E0a | 1e-6 | unattainable | unattainable | False |
| P3/64/E0b | 1e-3 | 2.377 | 2.179 | True |
| P3/64/E0b | 1e-4 | unattainable | unattainable | True |
| P3/64/E0b | 1e-5 | unattainable | unattainable | True |
| P3/64/E0b | 1e-6 | unattainable | unattainable | False |
| P4/64/E0a | 1e-3 | 0.4275 | 0.5194 | True |
| P4/64/E0a | 1e-4 | 0.3444 | 0.4206 | True |
| P4/64/E0a | 1e-5 | unattainable | unattainable | True |
| P4/64/E0a | 1e-6 | unattainable | unattainable | True |
| P4/64/E0b | 1e-3 | 1.237 | 1.13 | True |
| P4/64/E0b | 1e-4 | 1.175 | 1.076 | True |
| P4/64/E0b | 1e-5 | unattainable | unattainable | True |
| P4/64/E0b | 1e-6 | unattainable | unattainable | True |
| P5/64/E0a | 1e-3 | 1.847 | 2.257 | True |
| P5/64/E0a | 1e-4 | 0.9554 | 1.168 | True |
| P5/64/E0a | 1e-5 | unattainable | unattainable | False |
| P5/64/E0a | 1e-6 | unattainable | unattainable | False |
| P5/64/E0b | 1e-3 | 3.328 | 3.067 | True |
| P5/64/E0b | 1e-4 | 3.343 | 3.065 | True |
| P5/64/E0b | 1e-5 | unattainable | unattainable | False |
| P5/64/E0b | 1e-6 | unattainable | unattainable | False |
| P5/128/E0a | 1e-3 | 1.548 | 1.892 | True |
| P5/128/E0a | 1e-4 | 1.155 | 1.411 | True |
| P5/128/E0a | 1e-5 | unattainable | unattainable | True |
| P5/128/E0a | 1e-6 | unattainable | unattainable | False |
| P5/128/E0b | 1e-3 | 3.355 | 3.115 | True |
| P5/128/E0b | 1e-4 | 3.726 | 3.421 | True |
| P5/128/E0b | 1e-5 | unattainable | unattainable | True |
| P5/128/E0b | 1e-6 | unattainable | unattainable | False |

Source: `d3_work_precision.json`, `/matched_targets/*/*/Q_vs_S0_NFE`, `/matched_targets/*/*/Q_vs_S0_lean_NFE`, `/matched_targets/*/*/confirmed`.

## H3: fixed-step anchors (descriptive)

| Case / method / baseline | Median R |
| --- | ---: |
| P1/64/E0a/F | 0.3707 |
| P1/64/S0/F | 0.3335 |
| P2/128/E0a/F | 0.754 |
| P2/128/E0b/F | 1.677 |
| P2/128/S0/F | 0.8936 |
| P2/256/E0a/F | 0.717 |
| P2/256/E0b/F | 1.606 |
| P2/256/S0/F | 0.8487 |
| P3/64/E0a/F | 0.2293 |
| P3/64/E0b/F | 0.7978 |
| P3/64/S0/F | 0.3184 |
| P4/64/E0a/F | 0.1082 |
| P4/64/E0b/F | 0.3691 |
| P4/64/S0/F | 0.3301 |
| P5/64/E0a/F | 2.085 |
| P5/64/E0a/F_g | 0.6204 |
| P5/64/E0b/F | 3.756 |
| P5/64/E0b/F_g | 2.797 |
| P5/64/S0/F | 1.129 |
| P5/64/S0/F_g | 0.6652 |
| P5/128/E0a/F | 2.334 |
| P5/128/E0a/F_g | 0.6634 |
| P5/128/E0b/F | 5.059 |
| P5/128/E0b/F_g | 2.583 |
| P5/128/S0/F | 1.508 |
| P5/128/S0/F_g | 0.6258 |

Source: `d3_work_precision.json`, `/hypotheses/H3/*/median_R`.

## Estimator power and tolerance proportionality

| Case | Method | Time fraction | Median slope | Nominal power |
| --- | --- | ---: | ---: | ---: |
| P1/64 | S0 | 0.05 | 4.964 | 5 |
| P1/64 | E0a | 0.05 | 2.985 | 3 |
| P1/64 | S0 | 0.5 | 4.935 | 5 |
| P1/64 | E0a | 0.5 | 2.975 | 3 |
| P1/64 | S0 | 0.95 | 4.829 | 5 |
| P1/64 | E0a | 0.95 | 3.025 | 3 |
| P2/128 | S0 | 0.05 | 5.109 | 5 |
| P2/128 | E0a | 0.05 | 2.912 | 3 |
| P2/128 | E0b | 0.05 | 4.078 | 4 |
| P2/128 | S0 | 0.5 | 4.935 | 5 |
| P2/128 | E0a | 0.5 | 2.994 | 3 |
| P2/128 | E0b | 0.5 | 4.003 | 4 |
| P2/128 | S0 | 0.95 | 5.012 | 5 |
| P2/128 | E0a | 0.95 | 3.005 | 3 |
| P2/128 | E0b | 0.95 | 4.02 | 4 |
| P2/256 | S0 | 0.05 | 5.088 | 5 |
| P2/256 | E0a | 0.05 | 2.931 | 3 |
| P2/256 | E0b | 0.05 | 4.065 | 4 |
| P2/256 | S0 | 0.5 | 4.535 | 5 |
| P2/256 | E0a | 0.5 | 2.994 | 3 |
| P2/256 | E0b | 0.5 | 4.003 | 4 |
| P2/256 | S0 | 0.95 | 5.009 | 5 |
| P2/256 | E0a | 0.95 | 3.004 | 3 |
| P2/256 | E0b | 0.95 | 4.017 | 4 |
| P3/64 | S0 | 0.05 | 4.564 | 5 |
| P3/64 | E0a | 0.05 | 2.766 | 3 |
| P3/64 | E0b | 0.05 | 3.807 | 4 |
| P3/64 | S0 | 0.5 | 4.605 | 5 |
| P3/64 | E0a | 0.5 | 2.824 | 3 |
| P3/64 | E0b | 0.5 | 3.954 | 4 |
| P3/64 | S0 | 0.95 | 3.774 | 5 |
| P3/64 | E0a | 0.95 | 2.432 | 3 |
| P3/64 | E0b | 0.95 | 2.692 | 4 |
| P4/64 | S0 | 0.05 | 4.216 | 5 |
| P4/64 | E0a | 0.05 | 2.966 | 3 |
| P4/64 | E0b | 0.05 | 3.962 | 4 |
| P4/64 | S0 | 0.5 | 4.986 | 5 |
| P4/64 | E0a | 0.5 | 3.004 | 3 |
| P4/64 | E0b | 0.5 | 4.008 | 4 |
| P4/64 | S0 | 0.95 | 4.954 | 5 |
| P4/64 | E0a | 0.95 | 2.957 | 3 |
| P4/64 | E0b | 0.95 | 3.977 | 4 |
| P5/64 | S0 | 0.05 | 3.045 | 5 |
| P5/64 | E0a | 0.05 | 2.175 | 3 |
| P5/64 | E0b | 0.05 | 3.134 | 4 |
| P5/64 | S0 | 0.5 | 3.233 | 5 |
| P5/64 | E0a | 0.5 | 2.209 | 3 |
| P5/64 | E0b | 0.5 | 1.678 | 4 |
| P5/64 | S0 | 0.95 | 2.859 | 5 |
| P5/64 | E0a | 0.95 | 2.097 | 3 |
| P5/64 | E0b | 0.95 | 1.698 | 4 |
| P5/128 | S0 | 0.05 | 3.09 | 5 |
| P5/128 | E0a | 0.05 | 2.143 | 3 |
| P5/128 | E0b | 0.05 | 3.037 | 4 |
| P5/128 | S0 | 0.5 | 3.074 | 5 |
| P5/128 | E0a | 0.5 | 2.171 | 3 |
| P5/128 | E0b | 0.5 | 1.657 | 4 |
| P5/128 | S0 | 0.95 | 3.18 | 5 |
| P5/128 | E0a | 0.95 | 1.877 | 3 |
| P5/128 | E0b | 0.95 | 2.145 | 4 |

Source: `d3_work_precision.json`, `/estimator_power/*/entries/*/median_slope_k_ge_2`.

| Case / method | Tolerance slope | Valid records |
| --- | ---: | ---: |
| P1/64/E0a | 1.013 | 11 |
| P1/64/S0 | 0.7302 | 11 |
| P1/64/sensitivity | 0.6591 | 11 |
| P2/128/E0a | 0.9719 | 11 |
| P2/128/E0b | 0.954 | 11 |
| P2/128/S0 | 0.6986 | 11 |
| P2/128/sensitivity | 0.7229 | 9 |
| P2/256/E0a | 0.975 | 11 |
| P2/256/E0b | 0.9659 | 11 |
| P2/256/S0 | 0.7088 | 11 |
| P2/256/sensitivity | 0.7581 | 8 |
| P3/64/E0a | 1.015 | 6 |
| P3/64/E0b | 0.8896 | 7 |
| P3/64/S0 | 0.6802 | 5 |
| P3/64/sensitivity | 0.6529 | 5 |
| P4/64/E0a | 1.008 | 7 |
| P4/64/E0b | 0.8797 | 8 |
| P4/64/S0 | 0.6706 | 8 |
| P4/64/sensitivity | 0.6708 | 7 |
| P5/64/E0a | 0.7867 | 10 |
| P5/64/E0b | 0.855 | 11 |
| P5/64/S0 | 0.5025 | 11 |
| P5/64/sensitivity | 0.4676 | 10 |
| P5/128/E0a | 0.8391 | 9 |
| P5/128/E0b | 0.8275 | 9 |
| P5/128/S0 | 0.4804 | 10 |
| P5/128/sensitivity | 0.433 | 10 |

Source: `d3_work_precision.json`, `/tolerance_proportionality/*/slope`, `/tolerance_proportionality/*/n`.

## CPU timing ratios

| Case / method | Time ratio against S0 | Bootstrap interval |
| --- | ---: | ---: |
| P1/64/E0a | 1.01 | 0.9837 to 1.051 |
| P1/64/F | 10.22 | 9.215 to 10.84 |
| P2/128/E0a | 1.042 | 0.9822 to 1.091 |
| P2/128/E0b | 1.503 | 1.418 to 1.572 |
| P2/128/F | 2.472 | 2.266 to 2.597 |
| P2/256/E0a | 0.8807 | 0.8777 to 0.8847 |
| P2/256/E0b | 1.276 | 1.272 to 1.28 |
| P2/256/F | 3.306 | 3.281 to 3.325 |
| P4/64/E0a | 0.5688 | 0.4795 to 0.6606 |
| P4/64/E0b | 1.648 | 1.342 to 1.933 |
| P4/64/F | 9.388 | 7.879 to 10.9 |
| P5/64/E0a | 1.775 | 1.479 to 1.897 |
| P5/64/E0b | 5.794 | 4.842 to 6.202 |
| P5/128/E0a | 0.9793 | 0.8549 to 1.081 |
| P5/128/E0b | 2.853 | 2.562 to 3.163 |

Source: `d3_work_precision.json`, `/timing/ratios/*/Q_time_vs_S0`, `/timing/ratios/*/bootstrap_95pct`.

## Timing protocol and between-session check

Each timed record was warmed up and checked against its endpoint hash (`benchmarks/d3_work_precision_analysis.py:timing_case`). The runs used 5 shuffled blocks and up to 10 runs per block (`d3_timing_runs.csv`), then the median time was interpolated in log–log space at the matched 0.0001 (`/timing_plan/target`) target between bracketing Pareto records (`benchmarks/d3_work_precision_analysis.py:timed_summary`). Intervals use 2000 (`/timing/bootstrap_resamples`) bootstrap resamples of runs.

| Case / configuration | No-bracket reason |
| --- | --- |
| P1/64/E0b | method ineligible |
| P3/64/S0 | target outside valid Pareto error range |
| P5/64/F | target outside valid Pareto error range |
| P5/128/F | target outside valid Pareto error range |

Source: `d3_work_precision.json`, `/timing_plan/no_bracket/*/reason`.

| Case / pair | Study Q [interval] | This run Q [interval] | Overlap |
| --- | --- | --- | --- |
| P1/64/E0a | 0.9835 (`/timing/ratios/P1~164~1E0a/Q_time_vs_S0`) [0.946 (`/timing/ratios/P1~164~1E0a/bootstrap_95pct/0`), 1.014 (`/timing/ratios/P1~164~1E0a/bootstrap_95pct/1`)] | 1.01 (`/timing/ratios/P1~164~1E0a/Q_time_vs_S0`) [0.9837 (`/timing/ratios/P1~164~1E0a/bootstrap_95pct/0`), 1.051 (`/timing/ratios/P1~164~1E0a/bootstrap_95pct/1`)] | True |
| P1/64/F | 9.98 (`/timing/ratios/P1~164~1F/Q_time_vs_S0`) [9.286 (`/timing/ratios/P1~164~1F/bootstrap_95pct/0`), 10.4 (`/timing/ratios/P1~164~1F/bootstrap_95pct/1`)] | 10.22 (`/timing/ratios/P1~164~1F/Q_time_vs_S0`) [9.215 (`/timing/ratios/P1~164~1F/bootstrap_95pct/0`), 10.84 (`/timing/ratios/P1~164~1F/bootstrap_95pct/1`)] | True |
| P2/128/E0a | 0.9007 (`/timing/ratios/P2~1128~1E0a/Q_time_vs_S0`) [0.8643 (`/timing/ratios/P2~1128~1E0a/bootstrap_95pct/0`), 0.9376 (`/timing/ratios/P2~1128~1E0a/bootstrap_95pct/1`)] | 1.042 (`/timing/ratios/P2~1128~1E0a/Q_time_vs_S0`) [0.9822 (`/timing/ratios/P2~1128~1E0a/bootstrap_95pct/0`), 1.091 (`/timing/ratios/P2~1128~1E0a/bootstrap_95pct/1`)] | False |
| P2/128/E0b | 1.289 (`/timing/ratios/P2~1128~1E0b/Q_time_vs_S0`) [1.247 (`/timing/ratios/P2~1128~1E0b/bootstrap_95pct/0`), 1.339 (`/timing/ratios/P2~1128~1E0b/bootstrap_95pct/1`)] | 1.503 (`/timing/ratios/P2~1128~1E0b/Q_time_vs_S0`) [1.418 (`/timing/ratios/P2~1128~1E0b/bootstrap_95pct/0`), 1.572 (`/timing/ratios/P2~1128~1E0b/bootstrap_95pct/1`)] | False |
| P2/128/F | 2.154 (`/timing/ratios/P2~1128~1F/Q_time_vs_S0`) [2.083 (`/timing/ratios/P2~1128~1F/bootstrap_95pct/0`), 2.254 (`/timing/ratios/P2~1128~1F/bootstrap_95pct/1`)] | 2.472 (`/timing/ratios/P2~1128~1F/Q_time_vs_S0`) [2.266 (`/timing/ratios/P2~1128~1F/bootstrap_95pct/0`), 2.597 (`/timing/ratios/P2~1128~1F/bootstrap_95pct/1`)] | False |
| P2/256/E0a | 0.8734 (`/timing/ratios/P2~1256~1E0a/Q_time_vs_S0`) [0.8311 (`/timing/ratios/P2~1256~1E0a/bootstrap_95pct/0`), 0.9195 (`/timing/ratios/P2~1256~1E0a/bootstrap_95pct/1`)] | 0.8807 (`/timing/ratios/P2~1256~1E0a/Q_time_vs_S0`) [0.8777 (`/timing/ratios/P2~1256~1E0a/bootstrap_95pct/0`), 0.8847 (`/timing/ratios/P2~1256~1E0a/bootstrap_95pct/1`)] | True |
| P2/256/E0b | 1.262 (`/timing/ratios/P2~1256~1E0b/Q_time_vs_S0`) [1.225 (`/timing/ratios/P2~1256~1E0b/bootstrap_95pct/0`), 1.321 (`/timing/ratios/P2~1256~1E0b/bootstrap_95pct/1`)] | 1.276 (`/timing/ratios/P2~1256~1E0b/Q_time_vs_S0`) [1.272 (`/timing/ratios/P2~1256~1E0b/bootstrap_95pct/0`), 1.28 (`/timing/ratios/P2~1256~1E0b/bootstrap_95pct/1`)] | True |
| P2/256/F | 3.243 (`/timing/ratios/P2~1256~1F/Q_time_vs_S0`) [3.077 (`/timing/ratios/P2~1256~1F/bootstrap_95pct/0`), 3.394 (`/timing/ratios/P2~1256~1F/bootstrap_95pct/1`)] | 3.306 (`/timing/ratios/P2~1256~1F/Q_time_vs_S0`) [3.281 (`/timing/ratios/P2~1256~1F/bootstrap_95pct/0`), 3.325 (`/timing/ratios/P2~1256~1F/bootstrap_95pct/1`)] | True |
| P4/64/E0a | 0.3056 (`/timing/ratios/P4~164~1E0a/Q_time_vs_S0`) [0.3023 (`/timing/ratios/P4~164~1E0a/bootstrap_95pct/0`), 0.3136 (`/timing/ratios/P4~164~1E0a/bootstrap_95pct/1`)] | 0.5688 (`/timing/ratios/P4~164~1E0a/Q_time_vs_S0`) [0.4795 (`/timing/ratios/P4~164~1E0a/bootstrap_95pct/0`), 0.6606 (`/timing/ratios/P4~164~1E0a/bootstrap_95pct/1`)] | False |
| P4/64/E0b | 1.02 (`/timing/ratios/P4~164~1E0b/Q_time_vs_S0`) [1.007 (`/timing/ratios/P4~164~1E0b/bootstrap_95pct/0`), 1.046 (`/timing/ratios/P4~164~1E0b/bootstrap_95pct/1`)] | 1.648 (`/timing/ratios/P4~164~1E0b/Q_time_vs_S0`) [1.342 (`/timing/ratios/P4~164~1E0b/bootstrap_95pct/0`), 1.933 (`/timing/ratios/P4~164~1E0b/bootstrap_95pct/1`)] | False |
| P4/64/F | 5.242 (`/timing/ratios/P4~164~1F/Q_time_vs_S0`) [5.177 (`/timing/ratios/P4~164~1F/bootstrap_95pct/0`), 5.37 (`/timing/ratios/P4~164~1F/bootstrap_95pct/1`)] | 9.388 (`/timing/ratios/P4~164~1F/Q_time_vs_S0`) [7.879 (`/timing/ratios/P4~164~1F/bootstrap_95pct/0`), 10.9 (`/timing/ratios/P4~164~1F/bootstrap_95pct/1`)] | False |
| P5/64/E0a | 1.171 (`/timing/ratios/P5~164~1E0a/Q_time_vs_S0`) [1.064 (`/timing/ratios/P5~164~1E0a/bootstrap_95pct/0`), 1.225 (`/timing/ratios/P5~164~1E0a/bootstrap_95pct/1`)] | 1.775 (`/timing/ratios/P5~164~1E0a/Q_time_vs_S0`) [1.479 (`/timing/ratios/P5~164~1E0a/bootstrap_95pct/0`), 1.897 (`/timing/ratios/P5~164~1E0a/bootstrap_95pct/1`)] | False |
| P5/64/E0b | 3.91 (`/timing/ratios/P5~164~1E0b/Q_time_vs_S0`) [3.538 (`/timing/ratios/P5~164~1E0b/bootstrap_95pct/0`), 4.14 (`/timing/ratios/P5~164~1E0b/bootstrap_95pct/1`)] | 5.794 (`/timing/ratios/P5~164~1E0b/Q_time_vs_S0`) [4.842 (`/timing/ratios/P5~164~1E0b/bootstrap_95pct/0`), 6.202 (`/timing/ratios/P5~164~1E0b/bootstrap_95pct/1`)] | False |
| P5/128/E0a | 1.091 (`/timing/ratios/P5~1128~1E0a/Q_time_vs_S0`) [1.081 (`/timing/ratios/P5~1128~1E0a/bootstrap_95pct/0`), 1.104 (`/timing/ratios/P5~1128~1E0a/bootstrap_95pct/1`)] | 0.9793 (`/timing/ratios/P5~1128~1E0a/Q_time_vs_S0`) [0.8549 (`/timing/ratios/P5~1128~1E0a/bootstrap_95pct/0`), 1.081 (`/timing/ratios/P5~1128~1E0a/bootstrap_95pct/1`)] | True |
| P5/128/E0b | 3.115 (`/timing/ratios/P5~1128~1E0b/Q_time_vs_S0`) [3.085 (`/timing/ratios/P5~1128~1E0b/bootstrap_95pct/0`), 3.155 (`/timing/ratios/P5~1128~1E0b/bootstrap_95pct/1`)] | 2.853 (`/timing/ratios/P5~1128~1E0b/Q_time_vs_S0`) [2.562 (`/timing/ratios/P5~1128~1E0b/bootstrap_95pct/0`), 3.163 (`/timing/ratios/P5~1128~1E0b/bootstrap_95pct/1`)] | True |

Source: study `d3_timing_study_session.json` and this run `d3_work_precision.json`, `/timing/ratios/*`.

| Timed record ID | Study IQR/median | This run IQR/median |
| --- | ---: | ---: |
| P1_64_E0a_0.0001 | 0.0848 | 0.0783 |
| P1_64_E0a_0.0003 | 0.0975 | 0.0855 |
| P1_64_F_4 | 0.31 | 0.38 |
| P1_64_F_8 | 0.196 | 0.249 |
| P1_64_S0_0.0003 | 0.12 | 0.116 |
| P1_64_S0_0.001 | 0.103 | 0.12 |
| P2_128_E0a_0.0001 | 0.128 | 0.076 |
| P2_128_E0a_0.0003 | 0.107 | 0.0648 |
| P2_128_E0b_0.0001 | 0.128 | 0.0489 |
| P2_128_E0b_0.0003 | 0.0851 | 0.0474 |
| P2_128_F_128 | 0.11 | 0.81 |
| P2_128_F_64 | 0.12 | 0.585 |
| P2_128_S0_1e-05 | 0.089 | 0.288 |
| P2_128_S0_3e-05 | 0.12 | 0.213 |
| P2_256_E0a_0.0001 | 0.121 | 0.0971 |
| P2_256_E0a_0.0003 | 0.157 | 0.0164 |
| P2_256_E0b_0.0001 | 0.0959 | 0.0574 |
| P2_256_E0b_0.0003 | 0.0834 | 0.0122 |
| P2_256_F_128 | 0.161 | 0.038 |
| P2_256_F_64 | 0.217 | 0.0291 |
| P2_256_S0_1e-05 | 0.111 | 0.0149 |
| P2_256_S0_3e-05 | 0.129 | 0.013 |
| P3_64_E0a_1e-07 | 0.0875 | 0.0181 |
| P3_64_E0a_3e-07 | 0.0752 | 0.0219 |
| P3_64_E0b_1e-07 | 0.0774 | 0.767 |
| P3_64_E0b_3e-08 | 0.0866 | 0.582 |
| P3_64_F_1024 | 0.0424 | 0.018 |
| P3_64_F_2048 | 0.0449 | 0.0263 |
| P4_64_E0a_1e-07 | 0.0158 | 0.0393 |
| P4_64_E0a_3e-07 | 0.0178 | 0.0312 |
| P4_64_E0b_1e-07 | 0.0435 | 0.37 |
| P4_64_E0b_3e-07 | 0.0465 | 0.901 |
| P4_64_F_256 | 0.029 | 0.0697 |
| P4_64_F_512 | 0.0297 | 0.0595 |
| P4_64_S0_1e-08 | 0.111 | 0.481 |
| P4_64_S0_3e-08 | 0.0944 | 0.574 |
| P5_64_E0a_1e-05 | 0.144 | 0.0519 |
| P5_64_E0a_3e-06 | 0.153 | 0.0563 |
| P5_64_E0b_1e-06 | 0.163 | 0.0327 |
| P5_64_E0b_3e-06 | 0.169 | 0.0377 |
| P5_64_S0_1e-07 | 0.331 | 0.345 |
| P5_64_S0_3e-08 | 0.349 | 0.557 |
| P5_128_E0a_1e-05 | 0.00704 | 0.245 |
| P5_128_E0a_3e-06 | 0.00841 | 0.211 |
| P5_128_E0b_1e-06 | 0.0124 | 0.206 |
| P5_128_E0b_3e-06 | 0.014 | 0.146 |
| P5_128_S0_1e-07 | 0.0471 | 0.303 |
| P5_128_S0_3e-08 | 0.05 | 0.247 |

Source: study `d3_timing_study_session.json` and this run `d3_work_precision.json`, `/timing/records/*/IQR_seconds`, `/timing/records/*/median_seconds`.

Within-session intervals do not cover between-session differences on this laptop; CPU-time ratios are not evidence for H1. The NFE records are bitwise reproducible.

## Reproduction

The 357 (`/d3/records_identical`) of 357 (`/d3/records_compared`) records reproduce the [study](https://github.com/GVourvachakis/moljax/tree/f3a05c93a07a2cf3684660f94f410b989bd119e6) bitwise on the same laptop CPU and JAX 0.11.2. Other hosts can differ at rounding level, because the references are host specific.

Source: `d3_reproduction.json`, `/d3`; `d3_references.json`, `/host_specific`.

## Limits

ERK43ZB's 1/6 stage is outside the tick clock's one-rounding guarantee, so it is restricted to autonomous RHS. Measurements used a laptop CPU. Filter presets and the adaptive IMEX comparator remain on [experiment/etd-embedded-pairs](https://github.com/GVourvachakis/moljax/tree/f3a05c93a07a2cf3684660f94f410b989bd119e6); they are not part of this branch.

Source: `d3_work_precision.json`, `/settings`.
