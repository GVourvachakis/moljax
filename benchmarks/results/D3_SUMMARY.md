# D3: adaptive exponential work precision

## Setup and protocol

The H1 threshold and matched accuracy targets were fixed before comparison. The reference arrays and their checksums are in `d3_references.npz` and `d3_references.json`. CPU timings are secondary evidence.

H1 requires Q ≥ 1.15 (`/settings/H1_thresholds/support_Q`) on at least 3 (`/settings/H1_thresholds/minimum_problem_count`) eligible problems, with no problem below 0.9 (`/settings/H1_thresholds/loss_Q`).

| Case | Reference confirmation level | Method |
| --- | ---: | --- |
| P1_64 | 0 (`/reference_confirmation/P1_64/epsilon_conf`) | exact |
| P2_128 | 1.586e-09 (`/reference_confirmation/P2_128/epsilon_conf`) | self-convergence in D3 metric |
| P2_256 | 1.719e-09 (`/reference_confirmation/P2_256/epsilon_conf`) | self-convergence in D3 metric |
| P3_64 | 5.524e-07 (`/reference_confirmation/P3_64/epsilon_conf`) | endpoint distance plus bounded self-error |
| P4_64 | 2.877e-08 (`/reference_confirmation/P4_64/epsilon_conf`) | endpoint distance plus bounded self-error |
| P5_128 | 5.549e-07 (`/reference_confirmation/P5_128/epsilon_conf`) | endpoint distance plus bounded self-error |
| P5_64 | 9.065e-06 (`/reference_confirmation/P5_64/epsilon_conf`) | endpoint distance plus bounded self-error |

Source: `d3_work_precision.json`, `/settings`, `/reference_confirmation`, `/provenance`.

## H1: embedded pairs against step doubling

| Pair | Problem | Median Q | Supported |
| --- | --- | ---: | --- |
| E0a | P1 | 1.151 (`/hypotheses/H1/E0a/by_problem/P1/median_Q`) | False |
| E0a | P2 | 0.7416 (`/hypotheses/H1/E0a/by_problem/P2/median_Q`) | False |
| E0a | P3 | 0.7304 (`/hypotheses/H1/E0a/by_problem/P3/median_Q`) | False |
| E0a | P4 | 0.386 (`/hypotheses/H1/E0a/by_problem/P4/median_Q`) | False |
| E0a | P5 | 1.351 (`/hypotheses/H1/E0a/by_problem/P5/median_Q`) | False |
| E0b | P2 | 1.865 (`/hypotheses/H1/E0b/by_problem/P2/median_Q`) | True |
| E0b | P3 | 2.377 (`/hypotheses/H1/E0b/by_problem/P3/median_Q`) | True |
| E0b | P4 | 1.206 (`/hypotheses/H1/E0b/by_problem/P4/median_Q`) | True |
| E0b | P5 | 3.349 (`/hypotheses/H1/E0b/by_problem/P5/median_Q`) | True |

Source: `d3_work_precision.json`, `/hypotheses/H1`.

The primary Q uses nominal NFE. Lean NFE and CPU-time ratios are reported as secondary checks.

| Case | Pair | Target | Q | Lean Q | Confirmed |
| --- | --- | ---: | ---: | ---: | --- |
| P1/64 | E0a | 0.001 (`/matched_targets/P1/64/E0a/0/target`) | unattainable (`/matched_targets/P1/64/E0a/0/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P1/64/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P1/64 | E0a | 0.0001 (`/matched_targets/P1/64/E0a/1/target`) | 1.421 (`/matched_targets/P1/64/E0a/1/Q_vs_S0_NFE`) | 1.709 (`/matched_targets/P1/64/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P1/64 | E0a | 1e-05 (`/matched_targets/P1/64/E0a/2/target`) | 1.151 (`/matched_targets/P1/64/E0a/2/Q_vs_S0_NFE`) | 1.393 (`/matched_targets/P1/64/E0a/2/Q_vs_S0_lean_NFE`) | True |
| P1/64 | E0a | 1e-06 (`/matched_targets/P1/64/E0a/3/target`) | 0.8447 (`/matched_targets/P1/64/E0a/3/Q_vs_S0_NFE`) | 1.027 (`/matched_targets/P1/64/E0a/3/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0a | 0.001 (`/matched_targets/P2/128/E0a/0/target`) | unattainable (`/matched_targets/P2/128/E0a/0/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P2/128/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0a | 0.0001 (`/matched_targets/P2/128/E0a/1/target`) | 0.9601 (`/matched_targets/P2/128/E0a/1/Q_vs_S0_NFE`) | 1.16 (`/matched_targets/P2/128/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0a | 1e-05 (`/matched_targets/P2/128/E0a/2/target`) | 0.7428 (`/matched_targets/P2/128/E0a/2/Q_vs_S0_NFE`) | 0.9011 (`/matched_targets/P2/128/E0a/2/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0a | 1e-06 (`/matched_targets/P2/128/E0a/3/target`) | 0.5587 (`/matched_targets/P2/128/E0a/3/Q_vs_S0_NFE`) | 0.6812 (`/matched_targets/P2/128/E0a/3/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0b | 0.001 (`/matched_targets/P2/128/E0b/0/target`) | unattainable (`/matched_targets/P2/128/E0b/0/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P2/128/E0b/0/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0b | 0.0001 (`/matched_targets/P2/128/E0b/1/target`) | 1.911 (`/matched_targets/P2/128/E0b/1/Q_vs_S0_NFE`) | 1.762 (`/matched_targets/P2/128/E0b/1/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0b | 1e-05 (`/matched_targets/P2/128/E0b/2/target`) | 1.846 (`/matched_targets/P2/128/E0b/2/Q_vs_S0_NFE`) | 1.701 (`/matched_targets/P2/128/E0b/2/Q_vs_S0_lean_NFE`) | True |
| P2/128 | E0b | 1e-06 (`/matched_targets/P2/128/E0b/3/target`) | 1.877 (`/matched_targets/P2/128/E0b/3/Q_vs_S0_NFE`) | 1.725 (`/matched_targets/P2/128/E0b/3/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0a | 0.001 (`/matched_targets/P2/256/E0a/0/target`) | unattainable (`/matched_targets/P2/256/E0a/0/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P2/256/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0a | 0.0001 (`/matched_targets/P2/256/E0a/1/target`) | 0.961 (`/matched_targets/P2/256/E0a/1/Q_vs_S0_NFE`) | 1.161 (`/matched_targets/P2/256/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0a | 1e-05 (`/matched_targets/P2/256/E0a/2/target`) | 0.7404 (`/matched_targets/P2/256/E0a/2/Q_vs_S0_NFE`) | 0.8986 (`/matched_targets/P2/256/E0a/2/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0a | 1e-06 (`/matched_targets/P2/256/E0a/3/target`) | 0.549 (`/matched_targets/P2/256/E0a/3/Q_vs_S0_NFE`) | 0.6694 (`/matched_targets/P2/256/E0a/3/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0b | 0.001 (`/matched_targets/P2/256/E0b/0/target`) | unattainable (`/matched_targets/P2/256/E0b/0/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P2/256/E0b/0/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0b | 0.0001 (`/matched_targets/P2/256/E0b/1/target`) | 1.936 (`/matched_targets/P2/256/E0b/1/Q_vs_S0_NFE`) | 1.776 (`/matched_targets/P2/256/E0b/1/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0b | 1e-05 (`/matched_targets/P2/256/E0b/2/target`) | 1.852 (`/matched_targets/P2/256/E0b/2/Q_vs_S0_NFE`) | 1.706 (`/matched_targets/P2/256/E0b/2/Q_vs_S0_lean_NFE`) | True |
| P2/256 | E0b | 1e-06 (`/matched_targets/P2/256/E0b/3/target`) | 1.853 (`/matched_targets/P2/256/E0b/3/Q_vs_S0_NFE`) | 1.704 (`/matched_targets/P2/256/E0b/3/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0a | 0.001 (`/matched_targets/P3/64/E0a/0/target`) | 0.7304 (`/matched_targets/P3/64/E0a/0/Q_vs_S0_NFE`) | 0.8926 (`/matched_targets/P3/64/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0a | 0.0001 (`/matched_targets/P3/64/E0a/1/target`) | unattainable (`/matched_targets/P3/64/E0a/1/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P3/64/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0a | 1e-05 (`/matched_targets/P3/64/E0a/2/target`) | unattainable (`/matched_targets/P3/64/E0a/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P3/64/E0a/2/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0a | 1e-06 (`/matched_targets/P3/64/E0a/3/target`) | unattainable (`/matched_targets/P3/64/E0a/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P3/64/E0a/3/Q_vs_S0_lean_NFE`) | False |
| P3/64 | E0b | 0.001 (`/matched_targets/P3/64/E0b/0/target`) | 2.377 (`/matched_targets/P3/64/E0b/0/Q_vs_S0_NFE`) | 2.179 (`/matched_targets/P3/64/E0b/0/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0b | 0.0001 (`/matched_targets/P3/64/E0b/1/target`) | unattainable (`/matched_targets/P3/64/E0b/1/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P3/64/E0b/1/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0b | 1e-05 (`/matched_targets/P3/64/E0b/2/target`) | unattainable (`/matched_targets/P3/64/E0b/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P3/64/E0b/2/Q_vs_S0_lean_NFE`) | True |
| P3/64 | E0b | 1e-06 (`/matched_targets/P3/64/E0b/3/target`) | unattainable (`/matched_targets/P3/64/E0b/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P3/64/E0b/3/Q_vs_S0_lean_NFE`) | False |
| P4/64 | E0a | 0.001 (`/matched_targets/P4/64/E0a/0/target`) | 0.4275 (`/matched_targets/P4/64/E0a/0/Q_vs_S0_NFE`) | 0.5194 (`/matched_targets/P4/64/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0a | 0.0001 (`/matched_targets/P4/64/E0a/1/target`) | 0.3444 (`/matched_targets/P4/64/E0a/1/Q_vs_S0_NFE`) | 0.4206 (`/matched_targets/P4/64/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0a | 1e-05 (`/matched_targets/P4/64/E0a/2/target`) | unattainable (`/matched_targets/P4/64/E0a/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P4/64/E0a/2/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0a | 1e-06 (`/matched_targets/P4/64/E0a/3/target`) | unattainable (`/matched_targets/P4/64/E0a/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P4/64/E0a/3/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0b | 0.001 (`/matched_targets/P4/64/E0b/0/target`) | 1.237 (`/matched_targets/P4/64/E0b/0/Q_vs_S0_NFE`) | 1.13 (`/matched_targets/P4/64/E0b/0/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0b | 0.0001 (`/matched_targets/P4/64/E0b/1/target`) | 1.175 (`/matched_targets/P4/64/E0b/1/Q_vs_S0_NFE`) | 1.076 (`/matched_targets/P4/64/E0b/1/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0b | 1e-05 (`/matched_targets/P4/64/E0b/2/target`) | unattainable (`/matched_targets/P4/64/E0b/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P4/64/E0b/2/Q_vs_S0_lean_NFE`) | True |
| P4/64 | E0b | 1e-06 (`/matched_targets/P4/64/E0b/3/target`) | unattainable (`/matched_targets/P4/64/E0b/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P4/64/E0b/3/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0a | 0.001 (`/matched_targets/P5/128/E0a/0/target`) | 1.548 (`/matched_targets/P5/128/E0a/0/Q_vs_S0_NFE`) | 1.892 (`/matched_targets/P5/128/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0a | 0.0001 (`/matched_targets/P5/128/E0a/1/target`) | 1.155 (`/matched_targets/P5/128/E0a/1/Q_vs_S0_NFE`) | 1.411 (`/matched_targets/P5/128/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0a | 1e-05 (`/matched_targets/P5/128/E0a/2/target`) | unattainable (`/matched_targets/P5/128/E0a/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/128/E0a/2/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0a | 1e-06 (`/matched_targets/P5/128/E0a/3/target`) | unattainable (`/matched_targets/P5/128/E0a/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/128/E0a/3/Q_vs_S0_lean_NFE`) | False |
| P5/128 | E0b | 0.001 (`/matched_targets/P5/128/E0b/0/target`) | 3.355 (`/matched_targets/P5/128/E0b/0/Q_vs_S0_NFE`) | 3.115 (`/matched_targets/P5/128/E0b/0/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0b | 0.0001 (`/matched_targets/P5/128/E0b/1/target`) | 3.726 (`/matched_targets/P5/128/E0b/1/Q_vs_S0_NFE`) | 3.421 (`/matched_targets/P5/128/E0b/1/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0b | 1e-05 (`/matched_targets/P5/128/E0b/2/target`) | unattainable (`/matched_targets/P5/128/E0b/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/128/E0b/2/Q_vs_S0_lean_NFE`) | True |
| P5/128 | E0b | 1e-06 (`/matched_targets/P5/128/E0b/3/target`) | unattainable (`/matched_targets/P5/128/E0b/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/128/E0b/3/Q_vs_S0_lean_NFE`) | False |
| P5/64 | E0a | 0.001 (`/matched_targets/P5/64/E0a/0/target`) | 1.847 (`/matched_targets/P5/64/E0a/0/Q_vs_S0_NFE`) | 2.257 (`/matched_targets/P5/64/E0a/0/Q_vs_S0_lean_NFE`) | True |
| P5/64 | E0a | 0.0001 (`/matched_targets/P5/64/E0a/1/target`) | 0.9554 (`/matched_targets/P5/64/E0a/1/Q_vs_S0_NFE`) | 1.168 (`/matched_targets/P5/64/E0a/1/Q_vs_S0_lean_NFE`) | True |
| P5/64 | E0a | 1e-05 (`/matched_targets/P5/64/E0a/2/target`) | unattainable (`/matched_targets/P5/64/E0a/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/64/E0a/2/Q_vs_S0_lean_NFE`) | False |
| P5/64 | E0a | 1e-06 (`/matched_targets/P5/64/E0a/3/target`) | unattainable (`/matched_targets/P5/64/E0a/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/64/E0a/3/Q_vs_S0_lean_NFE`) | False |
| P5/64 | E0b | 0.001 (`/matched_targets/P5/64/E0b/0/target`) | 3.328 (`/matched_targets/P5/64/E0b/0/Q_vs_S0_NFE`) | 3.067 (`/matched_targets/P5/64/E0b/0/Q_vs_S0_lean_NFE`) | True |
| P5/64 | E0b | 0.0001 (`/matched_targets/P5/64/E0b/1/target`) | 3.343 (`/matched_targets/P5/64/E0b/1/Q_vs_S0_NFE`) | 3.065 (`/matched_targets/P5/64/E0b/1/Q_vs_S0_lean_NFE`) | True |
| P5/64 | E0b | 1e-05 (`/matched_targets/P5/64/E0b/2/target`) | unattainable (`/matched_targets/P5/64/E0b/2/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/64/E0b/2/Q_vs_S0_lean_NFE`) | False |
| P5/64 | E0b | 1e-06 (`/matched_targets/P5/64/E0b/3/target`) | unattainable (`/matched_targets/P5/64/E0b/3/Q_vs_S0_NFE`) | unattainable (`/matched_targets/P5/64/E0b/3/Q_vs_S0_lean_NFE`) | False |

Source: `d3_work_precision.json`, `/matched_targets`.

## H3: fixed-step anchors (descriptive)

| Case / method / baseline | Median R |
| --- | ---: |
| P1/64/E0a/F | 0.3707 (`/hypotheses/H3/P1/64/E0a/F/median_R`) |
| P1/64/S0/F | 0.3335 (`/hypotheses/H3/P1/64/S0/F/median_R`) |
| P2/128/E0a/F | 0.754 (`/hypotheses/H3/P2/128/E0a/F/median_R`) |
| P2/128/E0b/F | 1.677 (`/hypotheses/H3/P2/128/E0b/F/median_R`) |
| P2/128/S0/F | 0.8936 (`/hypotheses/H3/P2/128/S0/F/median_R`) |
| P2/256/E0a/F | 0.717 (`/hypotheses/H3/P2/256/E0a/F/median_R`) |
| P2/256/E0b/F | 1.606 (`/hypotheses/H3/P2/256/E0b/F/median_R`) |
| P2/256/S0/F | 0.8487 (`/hypotheses/H3/P2/256/S0/F/median_R`) |
| P3/64/E0a/F | 0.2293 (`/hypotheses/H3/P3/64/E0a/F/median_R`) |
| P3/64/E0b/F | 0.7978 (`/hypotheses/H3/P3/64/E0b/F/median_R`) |
| P3/64/S0/F | 0.3184 (`/hypotheses/H3/P3/64/S0/F/median_R`) |
| P4/64/E0a/F | 0.1082 (`/hypotheses/H3/P4/64/E0a/F/median_R`) |
| P4/64/E0b/F | 0.3691 (`/hypotheses/H3/P4/64/E0b/F/median_R`) |
| P4/64/S0/F | 0.3301 (`/hypotheses/H3/P4/64/S0/F/median_R`) |
| P5/128/E0a/F | 2.334 (`/hypotheses/H3/P5/128/E0a/F/median_R`) |
| P5/128/E0a/F_g | 0.6634 (`/hypotheses/H3/P5/128/E0a/F_g/median_R`) |
| P5/128/E0b/F | 5.059 (`/hypotheses/H3/P5/128/E0b/F/median_R`) |
| P5/128/E0b/F_g | 2.583 (`/hypotheses/H3/P5/128/E0b/F_g/median_R`) |
| P5/128/S0/F | 1.508 (`/hypotheses/H3/P5/128/S0/F/median_R`) |
| P5/128/S0/F_g | 0.6258 (`/hypotheses/H3/P5/128/S0/F_g/median_R`) |
| P5/64/E0a/F | 2.085 (`/hypotheses/H3/P5/64/E0a/F/median_R`) |
| P5/64/E0a/F_g | 0.6204 (`/hypotheses/H3/P5/64/E0a/F_g/median_R`) |
| P5/64/E0b/F | 3.756 (`/hypotheses/H3/P5/64/E0b/F/median_R`) |
| P5/64/E0b/F_g | 2.797 (`/hypotheses/H3/P5/64/E0b/F_g/median_R`) |
| P5/64/S0/F | 1.129 (`/hypotheses/H3/P5/64/S0/F/median_R`) |
| P5/64/S0/F_g | 0.6652 (`/hypotheses/H3/P5/64/S0/F_g/median_R`) |

Source: `d3_work_precision.json`, `/hypotheses/H3`.

## Estimator power and tolerance proportionality

| Case | Method | Time fraction | Median slope | Nominal power |
| --- | --- | ---: | ---: | ---: |
| P1/64 | S0 | 0.05 (`/estimator_power/0/entries/0/target_fraction`) | 4.964 (`/estimator_power/0/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/0/entries/0/nominal_power`) |
| P1/64 | E0a | 0.05 (`/estimator_power/0/entries/1/target_fraction`) | 2.985 (`/estimator_power/0/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/0/entries/1/nominal_power`) |
| P1/64 | S0 | 0.5 (`/estimator_power/0/entries/2/target_fraction`) | 4.935 (`/estimator_power/0/entries/2/median_slope_k_ge_2`) | 5 (`/estimator_power/0/entries/2/nominal_power`) |
| P1/64 | E0a | 0.5 (`/estimator_power/0/entries/3/target_fraction`) | 2.975 (`/estimator_power/0/entries/3/median_slope_k_ge_2`) | 3 (`/estimator_power/0/entries/3/nominal_power`) |
| P1/64 | S0 | 0.95 (`/estimator_power/0/entries/4/target_fraction`) | 4.829 (`/estimator_power/0/entries/4/median_slope_k_ge_2`) | 5 (`/estimator_power/0/entries/4/nominal_power`) |
| P1/64 | E0a | 0.95 (`/estimator_power/0/entries/5/target_fraction`) | 3.025 (`/estimator_power/0/entries/5/median_slope_k_ge_2`) | 3 (`/estimator_power/0/entries/5/nominal_power`) |
| P2/128 | S0 | 0.05 (`/estimator_power/1/entries/0/target_fraction`) | 5.109 (`/estimator_power/1/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/1/entries/0/nominal_power`) |
| P2/128 | E0a | 0.05 (`/estimator_power/1/entries/1/target_fraction`) | 2.912 (`/estimator_power/1/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/1/entries/1/nominal_power`) |
| P2/128 | E0b | 0.05 (`/estimator_power/1/entries/2/target_fraction`) | 4.078 (`/estimator_power/1/entries/2/median_slope_k_ge_2`) | 4 (`/estimator_power/1/entries/2/nominal_power`) |
| P2/128 | S0 | 0.5 (`/estimator_power/1/entries/3/target_fraction`) | 4.935 (`/estimator_power/1/entries/3/median_slope_k_ge_2`) | 5 (`/estimator_power/1/entries/3/nominal_power`) |
| P2/128 | E0a | 0.5 (`/estimator_power/1/entries/4/target_fraction`) | 2.994 (`/estimator_power/1/entries/4/median_slope_k_ge_2`) | 3 (`/estimator_power/1/entries/4/nominal_power`) |
| P2/128 | E0b | 0.5 (`/estimator_power/1/entries/5/target_fraction`) | 4.003 (`/estimator_power/1/entries/5/median_slope_k_ge_2`) | 4 (`/estimator_power/1/entries/5/nominal_power`) |
| P2/128 | S0 | 0.95 (`/estimator_power/1/entries/6/target_fraction`) | 5.012 (`/estimator_power/1/entries/6/median_slope_k_ge_2`) | 5 (`/estimator_power/1/entries/6/nominal_power`) |
| P2/128 | E0a | 0.95 (`/estimator_power/1/entries/7/target_fraction`) | 3.005 (`/estimator_power/1/entries/7/median_slope_k_ge_2`) | 3 (`/estimator_power/1/entries/7/nominal_power`) |
| P2/128 | E0b | 0.95 (`/estimator_power/1/entries/8/target_fraction`) | 4.02 (`/estimator_power/1/entries/8/median_slope_k_ge_2`) | 4 (`/estimator_power/1/entries/8/nominal_power`) |
| P2/256 | S0 | 0.05 (`/estimator_power/2/entries/0/target_fraction`) | 5.088 (`/estimator_power/2/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/2/entries/0/nominal_power`) |
| P2/256 | E0a | 0.05 (`/estimator_power/2/entries/1/target_fraction`) | 2.931 (`/estimator_power/2/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/2/entries/1/nominal_power`) |
| P2/256 | E0b | 0.05 (`/estimator_power/2/entries/2/target_fraction`) | 4.065 (`/estimator_power/2/entries/2/median_slope_k_ge_2`) | 4 (`/estimator_power/2/entries/2/nominal_power`) |
| P2/256 | S0 | 0.5 (`/estimator_power/2/entries/3/target_fraction`) | 4.535 (`/estimator_power/2/entries/3/median_slope_k_ge_2`) | 5 (`/estimator_power/2/entries/3/nominal_power`) |
| P2/256 | E0a | 0.5 (`/estimator_power/2/entries/4/target_fraction`) | 2.994 (`/estimator_power/2/entries/4/median_slope_k_ge_2`) | 3 (`/estimator_power/2/entries/4/nominal_power`) |
| P2/256 | E0b | 0.5 (`/estimator_power/2/entries/5/target_fraction`) | 4.003 (`/estimator_power/2/entries/5/median_slope_k_ge_2`) | 4 (`/estimator_power/2/entries/5/nominal_power`) |
| P2/256 | S0 | 0.95 (`/estimator_power/2/entries/6/target_fraction`) | 5.009 (`/estimator_power/2/entries/6/median_slope_k_ge_2`) | 5 (`/estimator_power/2/entries/6/nominal_power`) |
| P2/256 | E0a | 0.95 (`/estimator_power/2/entries/7/target_fraction`) | 3.004 (`/estimator_power/2/entries/7/median_slope_k_ge_2`) | 3 (`/estimator_power/2/entries/7/nominal_power`) |
| P2/256 | E0b | 0.95 (`/estimator_power/2/entries/8/target_fraction`) | 4.017 (`/estimator_power/2/entries/8/median_slope_k_ge_2`) | 4 (`/estimator_power/2/entries/8/nominal_power`) |
| P3/64 | S0 | 0.05 (`/estimator_power/3/entries/0/target_fraction`) | 4.564 (`/estimator_power/3/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/3/entries/0/nominal_power`) |
| P3/64 | E0a | 0.05 (`/estimator_power/3/entries/1/target_fraction`) | 2.766 (`/estimator_power/3/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/3/entries/1/nominal_power`) |
| P3/64 | E0b | 0.05 (`/estimator_power/3/entries/2/target_fraction`) | 3.807 (`/estimator_power/3/entries/2/median_slope_k_ge_2`) | 4 (`/estimator_power/3/entries/2/nominal_power`) |
| P3/64 | S0 | 0.5 (`/estimator_power/3/entries/3/target_fraction`) | 4.605 (`/estimator_power/3/entries/3/median_slope_k_ge_2`) | 5 (`/estimator_power/3/entries/3/nominal_power`) |
| P3/64 | E0a | 0.5 (`/estimator_power/3/entries/4/target_fraction`) | 2.824 (`/estimator_power/3/entries/4/median_slope_k_ge_2`) | 3 (`/estimator_power/3/entries/4/nominal_power`) |
| P3/64 | E0b | 0.5 (`/estimator_power/3/entries/5/target_fraction`) | 3.954 (`/estimator_power/3/entries/5/median_slope_k_ge_2`) | 4 (`/estimator_power/3/entries/5/nominal_power`) |
| P3/64 | S0 | 0.95 (`/estimator_power/3/entries/6/target_fraction`) | 3.774 (`/estimator_power/3/entries/6/median_slope_k_ge_2`) | 5 (`/estimator_power/3/entries/6/nominal_power`) |
| P3/64 | E0a | 0.95 (`/estimator_power/3/entries/7/target_fraction`) | 2.432 (`/estimator_power/3/entries/7/median_slope_k_ge_2`) | 3 (`/estimator_power/3/entries/7/nominal_power`) |
| P3/64 | E0b | 0.95 (`/estimator_power/3/entries/8/target_fraction`) | 2.692 (`/estimator_power/3/entries/8/median_slope_k_ge_2`) | 4 (`/estimator_power/3/entries/8/nominal_power`) |
| P4/64 | S0 | 0.05 (`/estimator_power/4/entries/0/target_fraction`) | 4.216 (`/estimator_power/4/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/4/entries/0/nominal_power`) |
| P4/64 | E0a | 0.05 (`/estimator_power/4/entries/1/target_fraction`) | 2.966 (`/estimator_power/4/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/4/entries/1/nominal_power`) |
| P4/64 | E0b | 0.05 (`/estimator_power/4/entries/2/target_fraction`) | 3.962 (`/estimator_power/4/entries/2/median_slope_k_ge_2`) | 4 (`/estimator_power/4/entries/2/nominal_power`) |
| P4/64 | S0 | 0.5 (`/estimator_power/4/entries/3/target_fraction`) | 4.986 (`/estimator_power/4/entries/3/median_slope_k_ge_2`) | 5 (`/estimator_power/4/entries/3/nominal_power`) |
| P4/64 | E0a | 0.5 (`/estimator_power/4/entries/4/target_fraction`) | 3.004 (`/estimator_power/4/entries/4/median_slope_k_ge_2`) | 3 (`/estimator_power/4/entries/4/nominal_power`) |
| P4/64 | E0b | 0.5 (`/estimator_power/4/entries/5/target_fraction`) | 4.008 (`/estimator_power/4/entries/5/median_slope_k_ge_2`) | 4 (`/estimator_power/4/entries/5/nominal_power`) |
| P4/64 | S0 | 0.95 (`/estimator_power/4/entries/6/target_fraction`) | 4.954 (`/estimator_power/4/entries/6/median_slope_k_ge_2`) | 5 (`/estimator_power/4/entries/6/nominal_power`) |
| P4/64 | E0a | 0.95 (`/estimator_power/4/entries/7/target_fraction`) | 2.957 (`/estimator_power/4/entries/7/median_slope_k_ge_2`) | 3 (`/estimator_power/4/entries/7/nominal_power`) |
| P4/64 | E0b | 0.95 (`/estimator_power/4/entries/8/target_fraction`) | 3.977 (`/estimator_power/4/entries/8/median_slope_k_ge_2`) | 4 (`/estimator_power/4/entries/8/nominal_power`) |
| P5/64 | S0 | 0.05 (`/estimator_power/5/entries/0/target_fraction`) | 3.045 (`/estimator_power/5/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/5/entries/0/nominal_power`) |
| P5/64 | E0a | 0.05 (`/estimator_power/5/entries/1/target_fraction`) | 2.175 (`/estimator_power/5/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/5/entries/1/nominal_power`) |
| P5/64 | E0b | 0.05 (`/estimator_power/5/entries/2/target_fraction`) | 3.134 (`/estimator_power/5/entries/2/median_slope_k_ge_2`) | 4 (`/estimator_power/5/entries/2/nominal_power`) |
| P5/64 | S0 | 0.5 (`/estimator_power/5/entries/3/target_fraction`) | 3.233 (`/estimator_power/5/entries/3/median_slope_k_ge_2`) | 5 (`/estimator_power/5/entries/3/nominal_power`) |
| P5/64 | E0a | 0.5 (`/estimator_power/5/entries/4/target_fraction`) | 2.209 (`/estimator_power/5/entries/4/median_slope_k_ge_2`) | 3 (`/estimator_power/5/entries/4/nominal_power`) |
| P5/64 | E0b | 0.5 (`/estimator_power/5/entries/5/target_fraction`) | 1.678 (`/estimator_power/5/entries/5/median_slope_k_ge_2`) | 4 (`/estimator_power/5/entries/5/nominal_power`) |
| P5/64 | S0 | 0.95 (`/estimator_power/5/entries/6/target_fraction`) | 2.859 (`/estimator_power/5/entries/6/median_slope_k_ge_2`) | 5 (`/estimator_power/5/entries/6/nominal_power`) |
| P5/64 | E0a | 0.95 (`/estimator_power/5/entries/7/target_fraction`) | 2.097 (`/estimator_power/5/entries/7/median_slope_k_ge_2`) | 3 (`/estimator_power/5/entries/7/nominal_power`) |
| P5/64 | E0b | 0.95 (`/estimator_power/5/entries/8/target_fraction`) | 1.698 (`/estimator_power/5/entries/8/median_slope_k_ge_2`) | 4 (`/estimator_power/5/entries/8/nominal_power`) |
| P5/128 | S0 | 0.05 (`/estimator_power/6/entries/0/target_fraction`) | 3.09 (`/estimator_power/6/entries/0/median_slope_k_ge_2`) | 5 (`/estimator_power/6/entries/0/nominal_power`) |
| P5/128 | E0a | 0.05 (`/estimator_power/6/entries/1/target_fraction`) | 2.143 (`/estimator_power/6/entries/1/median_slope_k_ge_2`) | 3 (`/estimator_power/6/entries/1/nominal_power`) |
| P5/128 | E0b | 0.05 (`/estimator_power/6/entries/2/target_fraction`) | 3.037 (`/estimator_power/6/entries/2/median_slope_k_ge_2`) | 4 (`/estimator_power/6/entries/2/nominal_power`) |
| P5/128 | S0 | 0.5 (`/estimator_power/6/entries/3/target_fraction`) | 3.074 (`/estimator_power/6/entries/3/median_slope_k_ge_2`) | 5 (`/estimator_power/6/entries/3/nominal_power`) |
| P5/128 | E0a | 0.5 (`/estimator_power/6/entries/4/target_fraction`) | 2.171 (`/estimator_power/6/entries/4/median_slope_k_ge_2`) | 3 (`/estimator_power/6/entries/4/nominal_power`) |
| P5/128 | E0b | 0.5 (`/estimator_power/6/entries/5/target_fraction`) | 1.657 (`/estimator_power/6/entries/5/median_slope_k_ge_2`) | 4 (`/estimator_power/6/entries/5/nominal_power`) |
| P5/128 | S0 | 0.95 (`/estimator_power/6/entries/6/target_fraction`) | 3.18 (`/estimator_power/6/entries/6/median_slope_k_ge_2`) | 5 (`/estimator_power/6/entries/6/nominal_power`) |
| P5/128 | E0a | 0.95 (`/estimator_power/6/entries/7/target_fraction`) | 1.877 (`/estimator_power/6/entries/7/median_slope_k_ge_2`) | 3 (`/estimator_power/6/entries/7/nominal_power`) |
| P5/128 | E0b | 0.95 (`/estimator_power/6/entries/8/target_fraction`) | 2.145 (`/estimator_power/6/entries/8/median_slope_k_ge_2`) | 4 (`/estimator_power/6/entries/8/nominal_power`) |

Source: `d3_work_precision.json`, `/estimator_power`.

| Case / method | Tolerance slope | Valid records |
| --- | ---: | ---: |
| P1/64/E0a | 1.013 (`/tolerance_proportionality/P1/64/E0a/slope`) | 11 (`/tolerance_proportionality/P1/64/E0a/n`) |
| P1/64/S0 | 0.7302 (`/tolerance_proportionality/P1/64/S0/slope`) | 11 (`/tolerance_proportionality/P1/64/S0/n`) |
| P1/64/sensitivity | 0.6591 (`/tolerance_proportionality/P1/64/sensitivity/slope`) | 11 (`/tolerance_proportionality/P1/64/sensitivity/n`) |
| P2/128/E0a | 0.9719 (`/tolerance_proportionality/P2/128/E0a/slope`) | 11 (`/tolerance_proportionality/P2/128/E0a/n`) |
| P2/128/E0b | 0.954 (`/tolerance_proportionality/P2/128/E0b/slope`) | 11 (`/tolerance_proportionality/P2/128/E0b/n`) |
| P2/128/S0 | 0.6986 (`/tolerance_proportionality/P2/128/S0/slope`) | 11 (`/tolerance_proportionality/P2/128/S0/n`) |
| P2/128/sensitivity | 0.7229 (`/tolerance_proportionality/P2/128/sensitivity/slope`) | 9 (`/tolerance_proportionality/P2/128/sensitivity/n`) |
| P2/256/E0a | 0.975 (`/tolerance_proportionality/P2/256/E0a/slope`) | 11 (`/tolerance_proportionality/P2/256/E0a/n`) |
| P2/256/E0b | 0.9659 (`/tolerance_proportionality/P2/256/E0b/slope`) | 11 (`/tolerance_proportionality/P2/256/E0b/n`) |
| P2/256/S0 | 0.7088 (`/tolerance_proportionality/P2/256/S0/slope`) | 11 (`/tolerance_proportionality/P2/256/S0/n`) |
| P2/256/sensitivity | 0.7581 (`/tolerance_proportionality/P2/256/sensitivity/slope`) | 8 (`/tolerance_proportionality/P2/256/sensitivity/n`) |
| P3/64/E0a | 1.015 (`/tolerance_proportionality/P3/64/E0a/slope`) | 6 (`/tolerance_proportionality/P3/64/E0a/n`) |
| P3/64/E0b | 0.8896 (`/tolerance_proportionality/P3/64/E0b/slope`) | 7 (`/tolerance_proportionality/P3/64/E0b/n`) |
| P3/64/S0 | 0.6802 (`/tolerance_proportionality/P3/64/S0/slope`) | 5 (`/tolerance_proportionality/P3/64/S0/n`) |
| P3/64/sensitivity | 0.6529 (`/tolerance_proportionality/P3/64/sensitivity/slope`) | 5 (`/tolerance_proportionality/P3/64/sensitivity/n`) |
| P4/64/E0a | 1.008 (`/tolerance_proportionality/P4/64/E0a/slope`) | 7 (`/tolerance_proportionality/P4/64/E0a/n`) |
| P4/64/E0b | 0.8797 (`/tolerance_proportionality/P4/64/E0b/slope`) | 8 (`/tolerance_proportionality/P4/64/E0b/n`) |
| P4/64/S0 | 0.6706 (`/tolerance_proportionality/P4/64/S0/slope`) | 8 (`/tolerance_proportionality/P4/64/S0/n`) |
| P4/64/sensitivity | 0.6708 (`/tolerance_proportionality/P4/64/sensitivity/slope`) | 7 (`/tolerance_proportionality/P4/64/sensitivity/n`) |
| P5/128/E0a | 0.8391 (`/tolerance_proportionality/P5/128/E0a/slope`) | 9 (`/tolerance_proportionality/P5/128/E0a/n`) |
| P5/128/E0b | 0.8275 (`/tolerance_proportionality/P5/128/E0b/slope`) | 9 (`/tolerance_proportionality/P5/128/E0b/n`) |
| P5/128/S0 | 0.4804 (`/tolerance_proportionality/P5/128/S0/slope`) | 10 (`/tolerance_proportionality/P5/128/S0/n`) |
| P5/128/sensitivity | 0.433 (`/tolerance_proportionality/P5/128/sensitivity/slope`) | 10 (`/tolerance_proportionality/P5/128/sensitivity/n`) |
| P5/64/E0a | 0.7867 (`/tolerance_proportionality/P5/64/E0a/slope`) | 10 (`/tolerance_proportionality/P5/64/E0a/n`) |
| P5/64/E0b | 0.855 (`/tolerance_proportionality/P5/64/E0b/slope`) | 11 (`/tolerance_proportionality/P5/64/E0b/n`) |
| P5/64/S0 | 0.5025 (`/tolerance_proportionality/P5/64/S0/slope`) | 11 (`/tolerance_proportionality/P5/64/S0/n`) |
| P5/64/sensitivity | 0.4676 (`/tolerance_proportionality/P5/64/sensitivity/slope`) | 10 (`/tolerance_proportionality/P5/64/sensitivity/n`) |

Source: `d3_work_precision.json`, `/tolerance_proportionality`.

## CPU timing ratios

| Case / method | Time ratio against S0 | Bootstrap interval |
| --- | ---: | ---: |
| P1/64/E0a | 1.01 (`/timing/ratios/P1/64/E0a/Q_time_vs_S0`) | 0.9837 (`/timing/ratios/P1/64/E0a/bootstrap_95pct/0`) to 1.051 (`/timing/ratios/P1/64/E0a/bootstrap_95pct/1`) |
| P1/64/F | 10.22 (`/timing/ratios/P1/64/F/Q_time_vs_S0`) | 9.215 (`/timing/ratios/P1/64/F/bootstrap_95pct/0`) to 10.84 (`/timing/ratios/P1/64/F/bootstrap_95pct/1`) |
| P2/128/E0a | 1.042 (`/timing/ratios/P2/128/E0a/Q_time_vs_S0`) | 0.9822 (`/timing/ratios/P2/128/E0a/bootstrap_95pct/0`) to 1.091 (`/timing/ratios/P2/128/E0a/bootstrap_95pct/1`) |
| P2/128/E0b | 1.503 (`/timing/ratios/P2/128/E0b/Q_time_vs_S0`) | 1.418 (`/timing/ratios/P2/128/E0b/bootstrap_95pct/0`) to 1.572 (`/timing/ratios/P2/128/E0b/bootstrap_95pct/1`) |
| P2/128/F | 2.472 (`/timing/ratios/P2/128/F/Q_time_vs_S0`) | 2.266 (`/timing/ratios/P2/128/F/bootstrap_95pct/0`) to 2.597 (`/timing/ratios/P2/128/F/bootstrap_95pct/1`) |
| P2/256/E0a | 0.8807 (`/timing/ratios/P2/256/E0a/Q_time_vs_S0`) | 0.8777 (`/timing/ratios/P2/256/E0a/bootstrap_95pct/0`) to 0.8847 (`/timing/ratios/P2/256/E0a/bootstrap_95pct/1`) |
| P2/256/E0b | 1.276 (`/timing/ratios/P2/256/E0b/Q_time_vs_S0`) | 1.272 (`/timing/ratios/P2/256/E0b/bootstrap_95pct/0`) to 1.28 (`/timing/ratios/P2/256/E0b/bootstrap_95pct/1`) |
| P2/256/F | 3.306 (`/timing/ratios/P2/256/F/Q_time_vs_S0`) | 3.281 (`/timing/ratios/P2/256/F/bootstrap_95pct/0`) to 3.325 (`/timing/ratios/P2/256/F/bootstrap_95pct/1`) |
| P4/64/E0a | 0.5688 (`/timing/ratios/P4/64/E0a/Q_time_vs_S0`) | 0.4795 (`/timing/ratios/P4/64/E0a/bootstrap_95pct/0`) to 0.6606 (`/timing/ratios/P4/64/E0a/bootstrap_95pct/1`) |
| P4/64/E0b | 1.648 (`/timing/ratios/P4/64/E0b/Q_time_vs_S0`) | 1.342 (`/timing/ratios/P4/64/E0b/bootstrap_95pct/0`) to 1.933 (`/timing/ratios/P4/64/E0b/bootstrap_95pct/1`) |
| P4/64/F | 9.388 (`/timing/ratios/P4/64/F/Q_time_vs_S0`) | 7.879 (`/timing/ratios/P4/64/F/bootstrap_95pct/0`) to 10.9 (`/timing/ratios/P4/64/F/bootstrap_95pct/1`) |
| P5/128/E0a | 0.9793 (`/timing/ratios/P5/128/E0a/Q_time_vs_S0`) | 0.8549 (`/timing/ratios/P5/128/E0a/bootstrap_95pct/0`) to 1.081 (`/timing/ratios/P5/128/E0a/bootstrap_95pct/1`) |
| P5/128/E0b | 2.853 (`/timing/ratios/P5/128/E0b/Q_time_vs_S0`) | 2.562 (`/timing/ratios/P5/128/E0b/bootstrap_95pct/0`) to 3.163 (`/timing/ratios/P5/128/E0b/bootstrap_95pct/1`) |
| P5/64/E0a | 1.775 (`/timing/ratios/P5/64/E0a/Q_time_vs_S0`) | 1.479 (`/timing/ratios/P5/64/E0a/bootstrap_95pct/0`) to 1.897 (`/timing/ratios/P5/64/E0a/bootstrap_95pct/1`) |
| P5/64/E0b | 5.794 (`/timing/ratios/P5/64/E0b/Q_time_vs_S0`) | 4.842 (`/timing/ratios/P5/64/E0b/bootstrap_95pct/0`) to 6.202 (`/timing/ratios/P5/64/E0b/bootstrap_95pct/1`) |

Source: `d3_work_precision.json`, `/timing`.

## Limits

ERK43ZB's 1/6 stage is outside the tick clock's one-rounding guarantee, so it is restricted to autonomous RHS. Measurements used a laptop CPU. Filter presets and the adaptive IMEX comparator remain on [experiment/etd-embedded-pairs](https://github.com/GVourvachakis/moljax/tree/experiment/etd-embedded-pairs); they are not part of this branch.

Source: `d3_work_precision.json`, `/settings`.
