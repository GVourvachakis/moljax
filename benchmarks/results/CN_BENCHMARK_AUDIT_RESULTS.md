# CN benchmark convergence audit

CPU-only convergence evidence. Times in the JSON are execution logs, not benchmark timings.
Every value below cites its JSON pointer in `cn_benchmark_audit.json`.

## M0: homogeneous error matrix

| Model | rho(G0) | max rho(Gk) | modes rho >= 1 | mean-field error max |
|---|---:|---:|---:|---:|
| brusselator | 0.02500000000000001 (`/m0/brusselator/rho_zero_mode`) | 0.02500000000000001 (`/m0/brusselator/rho_max`) | 0 (`/m0/brusselator/modes_rho_ge_1`) | 1.1102230246251565e-16 (`/m0/brusselator/mean_field_error_max_abs`) |
| gray_scott | 0.025 (`/m0/gray_scott/rho_zero_mode`) | 0.025 (`/m0/gray_scott/rho_max`) | 0 (`/m0/gray_scott/modes_rho_ge_1`) | 1.1102230246251565e-16 (`/m0/gray_scott/mean_field_error_max_abs`) |
| schnakenberg | 5.0 (`/m0/schnakenberg/rho_zero_mode`) | 5.0 (`/m0/schnakenberg/rho_max`) | 45 (`/m0/schnakenberg/modes_rho_ge_1`) | 6.315935428978668e-15 (`/m0/schnakenberg/mean_field_error_max_abs`) |

## M1 and M2: CN trajectories

| Arm | Model | Category | Steps | First non-finite | Iterations min/median/mean/max |
|---|---|---|---:|---:|---|
| m1 | brusselator | MARGINAL (`/m1/brusselator/category`) | 1000 (`/m1/brusselator/steps`) | None (`/m1/brusselator/first_nonfinite_step`) | 4 (`/m1/brusselator/iteration_stats/min`), 6.0 (`/m1/brusselator/iteration_stats/median`), 6.288 (`/m1/brusselator/iteration_stats/mean`), 15 (`/m1/brusselator/iteration_stats/max`) |
| m1 | gray_scott | SOLVES (`/m1/gray_scott/category`) | 20000 (`/m1/gray_scott/steps`) | None (`/m1/gray_scott/first_nonfinite_step`) | 1 (`/m1/gray_scott/iteration_stats/min`), 1.0 (`/m1/gray_scott/iteration_stats/median`), 1.10425 (`/m1/gray_scott/iteration_stats/mean`), 10 (`/m1/gray_scott/iteration_stats/max`) |
| m1 | schnakenberg | DOES NOT SOLVE (`/m1/schnakenberg/category`) | 10000 (`/m1/schnakenberg/steps`) | 1 (`/m1/schnakenberg/first_nonfinite_step`) | 1 (`/m1/schnakenberg/iteration_stats/min`), 1.0 (`/m1/schnakenberg/iteration_stats/median`), 1.0012 (`/m1/schnakenberg/iteration_stats/mean`), 13 (`/m1/schnakenberg/iteration_stats/max`) |
| m2 | brusselator | SOLVES (`/m2/brusselator/category`) | 1000 (`/m2/brusselator/steps`) | None (`/m2/brusselator/first_nonfinite_step`) | 3 (`/m2/brusselator/iteration_stats/min`), 4.0 (`/m2/brusselator/iteration_stats/median`), 3.966 (`/m2/brusselator/iteration_stats/mean`), 15 (`/m2/brusselator/iteration_stats/max`) |
| m2 | gray_scott | SOLVES (`/m2/gray_scott/category`) | 20000 (`/m2/gray_scott/steps`) | None (`/m2/gray_scott/first_nonfinite_step`) | 1 (`/m2/gray_scott/iteration_stats/min`), 1.0 (`/m2/gray_scott/iteration_stats/median`), 1.067 (`/m2/gray_scott/iteration_stats/mean`), 9 (`/m2/gray_scott/iteration_stats/max`) |
| m2 | schnakenberg | DOES NOT SOLVE (`/m2/schnakenberg/category`) | 10000 (`/m2/schnakenberg/steps`) | 2 (`/m2/schnakenberg/first_nonfinite_step`) | 1 (`/m2/schnakenberg/iteration_stats/min`), 1.0 (`/m2/schnakenberg/iteration_stats/median`), 1.0021 (`/m2/schnakenberg/iteration_stats/mean`), 15 (`/m2/schnakenberg/iteration_stats/max`) |

Per-step exit reason, lagged and post-update residual, and finiteness are in `/m1/*/records/*` and `/m2/*/records/*`.
The paper's Table 14 footnote uses approximately three NFE per CN step; the measured iteration distributions above test that assumption.

## M3: Schnakenberg first-iterate systems

| State | Source | Preconditioner | Left iterations | Right iterations | kappa2(P) | Arnoldi rho |
|---|---|---|---:|---:|---:|---:|
| 0 (`/m3/0/state_step`) | transient-only ETDRK4 (`/m3/0/state_source`) | fft | 101 (`/m3/0/preconditioners/fft/left/iterations`) | 63 (`/m3/0/preconditioners/fft/right/iterations`) | 258726.75761191687 (`/m3/0/preconditioners/fft/kappa_2_P`) | 5.000344456396726 (`/m3/0/preconditioners/fft/arnoldi_rho_estimate`) |
| 0 (`/m3/0/state_step`) | transient-only ETDRK4 (`/m3/0/state_source`) | mean_field | 7 (`/m3/0/preconditioners/mean_field/left/iterations`) | 4 (`/m3/0/preconditioners/mean_field/right/iterations`) | 13064227.037427224 (`/m3/0/preconditioners/mean_field/kappa_2_P`) | 0.12483022096408314 (`/m3/0/preconditioners/mean_field/arnoldi_rho_estimate`) |
| 0 (`/m3/0/state_step`) | identity | identity | 101 (`/m3/0/identity/iterations`) | same system | 1 | not evaluated |
| 20 (`/m3/1/state_step`) | transient-only ETDRK4 (`/m3/1/state_source`) | fft | 98 (`/m3/1/preconditioners/fft/left/iterations`) | 101 (`/m3/1/preconditioners/fft/right/iterations`) | 258726.75761191687 (`/m3/1/preconditioners/fft/kappa_2_P`) | 5.0000591815803235 (`/m3/1/preconditioners/fft/arnoldi_rho_estimate`) |
| 20 (`/m3/1/state_step`) | transient-only ETDRK4 (`/m3/1/state_source`) | mean_field | 7 (`/m3/1/preconditioners/mean_field/left/iterations`) | 7 (`/m3/1/preconditioners/mean_field/right/iterations`) | 13173419.880617881 (`/m3/1/preconditioners/mean_field/kappa_2_P`) | 0.15520535204642166 (`/m3/1/preconditioners/mean_field/arnoldi_rho_estimate`) |
| 20 (`/m3/1/state_step`) | identity | identity | 101 (`/m3/1/identity/iterations`) | same system | 1 | not evaluated |
| 200 (`/m3/2/state_step`) | transient-only ETDRK4 (`/m3/2/state_source`) | fft | 101 (`/m3/2/preconditioners/fft/left/iterations`) | 101 (`/m3/2/preconditioners/fft/right/iterations`) | 258726.75761191687 (`/m3/2/preconditioners/fft/kappa_2_P`) | 6.779366251274716 (`/m3/2/preconditioners/fft/arnoldi_rho_estimate`) |
| 200 (`/m3/2/state_step`) | transient-only ETDRK4 (`/m3/2/state_source`) | mean_field | 101 (`/m3/2/preconditioners/mean_field/left/iterations`) | 101 (`/m3/2/preconditioners/mean_field/right/iterations`) | 116701.1197244397 (`/m3/2/preconditioners/mean_field/kappa_2_P`) | 1.0222139464176498 (`/m3/2/preconditioners/mean_field/arnoldi_rho_estimate`) |
| 200 (`/m3/2/state_step`) | identity | identity | 101 (`/m3/2/identity/iterations`) | same system | 1 | not evaluated |
| 2000 (`/m3/3/state_step`) | transient-only ETDRK4 (`/m3/3/state_source`) | fft | 32 (`/m3/3/preconditioners/fft/left/iterations`) | 35 (`/m3/3/preconditioners/fft/right/iterations`) | 258726.75761191687 (`/m3/3/preconditioners/fft/kappa_2_P`) | 6.779454566948491 (`/m3/3/preconditioners/fft/arnoldi_rho_estimate`) |
| 2000 (`/m3/3/state_step`) | transient-only ETDRK4 (`/m3/3/state_source`) | mean_field | 24 (`/m3/3/preconditioners/mean_field/left/iterations`) | 25 (`/m3/3/preconditioners/mean_field/right/iterations`) | 116661.39441022955 (`/m3/3/preconditioners/mean_field/kappa_2_P`) | 1.0219054636382767 (`/m3/3/preconditioners/mean_field/arnoldi_rho_estimate`) |
| 2000 (`/m3/3/state_step`) | identity | identity | 101 (`/m3/3/identity/iterations`) | same system | 1 | not evaluated |

State paths and SHA-256 hashes are recorded at `/m3/*/state`; binary state files remain outside Git.
