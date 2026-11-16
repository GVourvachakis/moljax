# D3 adaptive ETD work-precision study

The frozen v3 protocol was followed by descriptive v3.1 additions and the v3.2 cost-plan revision before this sweep. The first cost gate exposed equal-tolerance counts at rtol 1e-8 without errors. No protocol threshold, target, tolerance, configuration or primary metric changed in response.

All measurements used CPU and JAX 0.11.2. NFE is the primary work metric; timings are secondary.

## Provenance and gates

- Branch: `ea460b274f0907463013759f50015724a02d6478`; Python 3.12.3; JAX 0.11.2; jaxlib 0.11.2.
Source: `benchmarks/results/d3_work_precision.json`, `/provenance`.

- Stage A projection: 18.1 h against 24 h; Stage B price: 1.86 h against 8 h. Fallback: False.
Source: `benchmarks/results/d3_work_precision.json`, `/gates`.

- Earlier combined cost estimate: 31.7 h; revised Stage A planning upper bound: 11.4 h. These were planning estimates, not D3 results.
Source: `benchmarks/results/d3_work_precision.json`, `/gates/first_combined_projection_hours` and `/gates/stage_a_projection/stage_a_upper_hours`.

## Reference confirmation

| Case | epsilon_conf | Basis | Confirmed targets |
| --- | ---: | --- | --- |
| P1/64 | 0 | exact | 1e-03, 1e-04, 1e-05, 1e-06 |
| P2/128 | 1.586e-09 | self-convergence in D3 metric | 1e-03, 1e-04, 1e-05, 1e-06 |
| P2/256 | 1.719e-09 | self-convergence in D3 metric | 1e-03, 1e-04, 1e-05, 1e-06 |
| P3/64 | 5.524e-07 | recomputed relative endpoint distance plus bounded self-error | 1e-03, 1e-04, 1e-05 |
| P4/64 | 2.877e-08 | recomputed relative endpoint distance plus bounded self-error | 1e-03, 1e-04, 1e-05, 1e-06 |
| P5/64 | 9.065e-06 | recomputed relative endpoint distance plus bounded self-error | 1e-03, 1e-04 |
| P5/128 | 5.549e-07 | recomputed relative endpoint distance plus bounded self-error | 1e-03, 1e-04, 1e-05 |
Source: `benchmarks/results/d3_work_precision.json`, `/reference_confirmation` (rows in case order).

D0b validity rule, verbatim: `error_inf >= 100*est_err -> valid; 10*est_err <= error_inf < 100*est_err -> valid, reference-limited; error_inf < 10*est_err -> indeterminate`. This rule uses absolute error. Matched accuracy uses the maximum per-field relative infinity error.
Source: `benchmarks/results/d3_work_precision.json`, `/validity_rule` and `/settings/error_metric`.

## Stage A records

| Case | Record counts by configuration | Excluded |
| --- | --- | ---: |
| P1/64 | E0a 11, F 6, S0 11, S1a 11, S1b 11, S1c 11, sensitivity 11 | 0 |
| P2/128 | E0a 11, E0b 11, E1 11, F 7, S0 11, S1a 11, S1b 11, S1c 11, X 11, sensitivity 11 | 4 |
| P2/256 | E0a 11, E0b 11, E1 11, F 6, S0 11, S1a 11, S1b 11, S1c 11, X 11, sensitivity 11 | 5 |
| P3/64 | E0a 11, E0b 11, E1 11, F 8, S0 11, S1a 11, S1b 11, S1c 11, X 11, sensitivity 11 | 47 |
| P4/64 | E0a 11, E0b 11, E1 11, F 7, S0 11, S1a 11, S1b 11, S1c 11, X 11, sensitivity 11 | 28 |
| P5/128 | E0a 11, E0b 11, E1 11, F 8, F_g 5, S0 11, S1a 11, S1b 11, S1c 11, X 11, sensitivity 11 | 14 |
| P5/64 | E0a 11, E0b 11, E1 11, F 8, F_g 5, S0 11, S1a 11, S1b 11, S1c 11, X 11, sensitivity 11 | 6 |
Source: `benchmarks/results/d3_work_precision.json`, `/stage_a/counts` and `/stage_a/excluded_counts`.

| Failed case | Configuration | Tolerance/step | Status |
| --- | --- | --- | --- |
| P2/128 | F | 4 | non-finite |
| P2/128 | F | 8 | non-finite |
| P2/256 | F | 8 | non-finite |
| P3/64 | F | 16 | non-finite |
| P3/64 | F | 32 | non-finite |
| P3/64 | F | 64 | non-finite |
| P4/64 | F | 16 | non-finite |
| P4/64 | F | 32 | non-finite |
| P4/64 | F | 64 | non-finite |
| P5/64 | F | 128 | non-finite |
| P5/64 | F | 256 | non-finite |
| P5/64 | F | 512 | non-finite |
| P5/64 | X | 3e-05 | 2 |
| P5/64 | X | 1e-05 | 2 |
| P5/64 | X | 3e-06 | 2 |
| P5/64 | X | 1e-06 | 2 |
| P5/64 | X | 3e-07 | 2 |
| P5/64 | X | 1e-07 | 2 |
| P5/64 | X | 3e-08 | 2 |
| P5/64 | X | 1e-08 | 2 |
| P5/128 | F | 128 | non-finite |
| P5/128 | F | 256 | non-finite |
| P5/128 | F | 512 | non-finite |
| P5/128 | X | 0.0001 | not run (time) |
| P5/128 | X | 3e-05 | not run (time) |
| P5/128 | X | 1e-05 | not run (time) |
| P5/128 | X | 3e-06 | not run (time) |
| P5/128 | X | 1e-06 | not run (time) |
| P5/128 | X | 3e-07 | not run (time) |
| P5/128 | X | 1e-07 | not run (time) |
| P5/128 | X | 3e-08 | not run (time) |
| P5/128 | X | 1e-08 | not run (time) |
Source: `benchmarks/results/d3_work_precision.json`, `/stage_a/failures`.

## Matched accuracy and hypotheses

Q(A versus B) is B's interpolated NFE divided by A's at the same error; Q above one favours A. The tables below contain only targets in the recorded Pareto range.

| Pair | Problem | Median Q against S0 | H1 |
| --- | --- | ---: | --- |
| E0a | P1 | 1.151 | False |
| E0a | P2 | 0.7416 | False |
| E0a | P3 | 0.7304 | False |
| E0a | P4 | 0.386 | False |
| E0a | P5 | 1.351 | False |
| E0b | P2 | 1.865 | True |
| E0b | P3 | 2.377 | True |
| E0b | P4 | 1.206 | True |
| E0b | P5 | 3.349 | True |
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H1`.

| Filter | P4/64 Q | P5/64 Q | P5/128 Q | H2 |
| --- | ---: | ---: | ---: | --- |
| S1a | unavailable | unavailable | unavailable | neutral |
| S1b | unavailable | unavailable | unavailable | neutral |
| S1c | unavailable | unavailable | unavailable | neutral |
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H2`.

The H2 focus cases have no matched target in the prescribed tolerance window. The frozen rule returns neutral here; that is an unassessed comparison, not evidence of equal cost.
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H2`.

### Per-target work ratios

| Case | Configuration | Target | Q versus S0 | Lean-NFE Q | Confirmed target |
| --- | --- | ---: | ---: | ---: | --- |
| P1/64 | E0a | 1e-04 | 1.421 | 1.709 | True |
| P1/64 | E0a | 1e-05 | 1.151 | 1.393 | True |
| P1/64 | E0a | 1e-06 | 0.8447 | 1.027 | True |
| P1/64 | S1a | 1e-05 | 0.8291 | 0.8291 | True |
| P1/64 | S1a | 1e-06 | 0.817 | 0.817 | True |
| P1/64 | S1b | 1e-05 | 0.9208 | 0.9208 | True |
| P1/64 | S1b | 1e-06 | 0.9606 | 0.9606 | True |
| P1/64 | S1c | 1e-05 | 0.7868 | 0.7868 | True |
| P1/64 | S1c | 1e-06 | 0.8702 | 0.8702 | True |
| P1/64 | sensitivity | 1e-05 | 0.9869 | 0.9869 | True |
| P1/64 | sensitivity | 1e-06 | 0.9734 | 0.9734 | True |
| P2/128 | E0a | 1e-04 | 0.9601 | 1.16 | True |
| P2/128 | E0a | 1e-05 | 0.7428 | 0.9011 | True |
| P2/128 | E0a | 1e-06 | 0.5587 | 0.6812 | True |
| P2/128 | E0b | 1e-04 | 1.911 | 1.762 | True |
| P2/128 | E0b | 1e-05 | 1.846 | 1.701 | True |
| P2/128 | E0b | 1e-06 | 1.877 | 1.725 | True |
| P2/128 | E1 | 1e-04 | 1.911 | 1.762 | True |
| P2/128 | E1 | 1e-05 | 1.846 | 1.701 | True |
| P2/128 | E1 | 1e-06 | 1.877 | 1.725 | True |
| P2/128 | S1a | 1e-04 | 0.9375 | 0.9289 | True |
| P2/128 | S1a | 1e-05 | 0.932 | 0.9264 | True |
| P2/128 | S1a | 1e-06 | 0.9587 | 0.9569 | True |
| P2/128 | S1b | 1e-04 | 0.9942 | 0.9896 | True |
| P2/128 | S1b | 1e-05 | 0.9381 | 0.9338 | True |
| P2/128 | S1b | 1e-06 | 0.9928 | 0.991 | True |
| P2/128 | S1c | 1e-04 | 0.8939 | 0.8894 | True |
| P2/128 | S1c | 1e-05 | 0.8662 | 0.8633 | True |
| P2/128 | S1c | 1e-06 | 0.8935 | 0.8919 | True |
| P2/128 | X | 1e-03 | 0.2772 | n/a | True |
| P2/128 | X | 1e-04 | 0.1624 | n/a | True |
| P2/128 | X | 1e-05 | 0.07922 | n/a | True |
| P2/128 | sensitivity | 1e-03 | 0.7795 | 0.7779 | True |
| P2/128 | sensitivity | 1e-04 | 0.8001 | 0.7957 | True |
| P2/128 | sensitivity | 1e-05 | 0.803 | 0.7991 | True |
| P2/256 | E0a | 1e-04 | 0.961 | 1.161 | True |
| P2/256 | E0a | 1e-05 | 0.7404 | 0.8986 | True |
| P2/256 | E0a | 1e-06 | 0.549 | 0.6694 | True |
| P2/256 | E0b | 1e-04 | 1.936 | 1.776 | True |
| P2/256 | E0b | 1e-05 | 1.852 | 1.706 | True |
| P2/256 | E0b | 1e-06 | 1.853 | 1.704 | True |
| P2/256 | E1 | 1e-04 | 1.936 | 1.776 | True |
| P2/256 | E1 | 1e-05 | 1.852 | 1.706 | True |
| P2/256 | E1 | 1e-06 | 1.853 | 1.704 | True |
| P2/256 | S1a | 1e-03 | 0.9305 | 0.9201 | True |
| P2/256 | S1a | 1e-04 | 0.9402 | 0.9319 | True |
| P2/256 | S1a | 1e-05 | 0.9285 | 0.9233 | True |
| P2/256 | S1a | 1e-06 | 0.9542 | 0.9525 | True |
| P2/256 | S1b | 1e-04 | 0.944 | 0.9396 | True |
| P2/256 | S1b | 1e-05 | 0.9358 | 0.9317 | True |
| P2/256 | S1b | 1e-06 | 0.9847 | 0.9829 | True |
| P2/256 | S1c | 1e-04 | 0.8181 | 0.8139 | True |
| P2/256 | S1c | 1e-05 | 0.8507 | 0.8479 | True |
| P2/256 | S1c | 1e-06 | 0.8745 | 0.873 | True |
| P2/256 | X | 1e-03 | 0.3073 | n/a | True |
| P2/256 | X | 1e-04 | 0.1573 | n/a | True |
| P2/256 | X | 1e-05 | 0.07781 | n/a | True |
| P2/256 | sensitivity | 1e-03 | 0.8703 | 0.8642 | True |
| P2/256 | sensitivity | 1e-04 | 0.8042 | 0.8 | True |
| P2/256 | sensitivity | 1e-05 | 0.8003 | 0.7963 | True |
| P3/64 | E0a | 1e-03 | 0.7304 | 0.8926 | True |
| P3/64 | E0b | 1e-03 | 2.377 | 2.179 | True |
| P3/64 | E1 | 1e-03 | 2.377 | 2.179 | True |
| P3/64 | S1a | 1e-03 | 1.024 | 1.024 | True |
| P3/64 | S1b | 1e-03 | 1.001 | 1.001 | True |
| P3/64 | S1c | 1e-03 | 0.9951 | 0.9951 | True |
| P3/64 | X | 1e-03 | 0.01413 | n/a | True |
| P3/64 | sensitivity | 1e-03 | 0.9819 | 0.9819 | True |
| P4/64 | E0a | 1e-03 | 0.4275 | 0.5194 | True |
| P4/64 | E0a | 1e-04 | 0.3444 | 0.4206 | True |
| P4/64 | E0b | 1e-03 | 1.237 | 1.13 | True |
| P4/64 | E0b | 1e-04 | 1.175 | 1.076 | True |
| P4/64 | E1 | 1e-03 | 1.237 | 1.13 | True |
| P4/64 | E1 | 1e-04 | 1.175 | 1.076 | True |
| P4/64 | S1a | 1e-03 | 1.265 | 1.26 | True |
| P4/64 | S1a | 1e-04 | 1.12 | 1.119 | True |
| P4/64 | S1b | 1e-03 | 1.122 | 1.118 | True |
| P4/64 | S1b | 1e-04 | 1.045 | 1.045 | True |
| P4/64 | S1c | 1e-03 | 1.055 | 1.051 | True |
| P4/64 | S1c | 1e-04 | 1.033 | 1.032 | True |
| P4/64 | X | 1e-03 | 0.08951 | n/a | True |
| P4/64 | sensitivity | 1e-03 | 1.008 | 1.007 | True |
| P4/64 | sensitivity | 1e-04 | 1.012 | 1.011 | True |
| P5/128 | E0a | 1e-03 | 1.548 | 1.892 | True |
| P5/128 | E0a | 1e-04 | 1.155 | 1.411 | True |
| P5/128 | E0b | 1e-03 | 3.355 | 3.115 | True |
| P5/128 | E0b | 1e-04 | 3.726 | 3.421 | True |
| P5/128 | E1 | 1e-03 | 3.355 | 3.115 | True |
| P5/128 | E1 | 1e-04 | 3.726 | 3.421 | True |
| P5/128 | S1a | 1e-03 | 1.037 | 1.037 | True |
| P5/128 | S1a | 1e-04 | 1.037 | 1.037 | True |
| P5/128 | S1b | 1e-03 | 0.971 | 0.9709 | True |
| P5/128 | S1b | 1e-04 | 1.019 | 1.019 | True |
| P5/128 | S1c | 1e-03 | 0.946 | 0.9459 | True |
| P5/128 | S1c | 1e-04 | 0.996 | 0.996 | True |
| P5/128 | sensitivity | 1e-03 | 1.035 | 1.035 | True |
| P5/128 | sensitivity | 1e-04 | 1.063 | 1.063 | True |
| P5/64 | E0a | 1e-03 | 1.847 | 2.257 | True |
| P5/64 | E0a | 1e-04 | 0.9554 | 1.168 | True |
| P5/64 | E0b | 1e-03 | 3.328 | 3.067 | True |
| P5/64 | E0b | 1e-04 | 3.343 | 3.065 | True |
| P5/64 | E1 | 1e-03 | 3.328 | 3.067 | True |
| P5/64 | E1 | 1e-04 | 3.343 | 3.065 | True |
| P5/64 | S1a | 1e-03 | 0.9987 | 0.9987 | True |
| P5/64 | S1a | 1e-04 | 1.006 | 1.006 | True |
| P5/64 | S1b | 1e-03 | 1.015 | 1.015 | True |
| P5/64 | S1b | 1e-04 | 0.9976 | 0.9976 | True |
| P5/64 | S1c | 1e-03 | 1.005 | 1.005 | True |
| P5/64 | S1c | 1e-04 | 1.01 | 1.01 | True |
| P5/64 | sensitivity | 1e-03 | 1.072 | 1.072 | True |
| P5/64 | sensitivity | 1e-04 | 1.034 | 1.034 | True |
Source: `benchmarks/results/d3_work_precision.json`, `/matched_targets`.

| Case | Configuration | Attainable targets |
| --- | --- | --- |
| P1/64 | E0a | 1e-04, 1e-05, 1e-06 |
| P1/64 | S0 | 1e-04, 1e-05, 1e-06 |
| P1/64 | S1a | 1e-05, 1e-06 |
| P1/64 | S1b | 1e-05, 1e-06 |
| P1/64 | S1c | 1e-05, 1e-06 |
| P1/64 | sensitivity | 1e-05, 1e-06 |
| P2/128 | E0a | 1e-04, 1e-05, 1e-06 |
| P2/128 | E0b | 1e-04, 1e-05, 1e-06 |
| P2/128 | E1 | 1e-04, 1e-05, 1e-06 |
| P2/128 | S0 | 1e-03, 1e-04, 1e-05, 1e-06 |
| P2/128 | S1a | 1e-04, 1e-05, 1e-06 |
| P2/128 | S1b | 1e-04, 1e-05, 1e-06 |
| P2/128 | S1c | 1e-04, 1e-05, 1e-06 |
| P2/128 | X | 1e-03, 1e-04, 1e-05 |
| P2/128 | sensitivity | 1e-03, 1e-04, 1e-05 |
| P2/256 | E0a | 1e-04, 1e-05, 1e-06 |
| P2/256 | E0b | 1e-04, 1e-05, 1e-06 |
| P2/256 | E1 | 1e-04, 1e-05, 1e-06 |
| P2/256 | S0 | 1e-03, 1e-04, 1e-05, 1e-06 |
| P2/256 | S1a | 1e-03, 1e-04, 1e-05, 1e-06 |
| P2/256 | S1b | 1e-04, 1e-05, 1e-06 |
| P2/256 | S1c | 1e-04, 1e-05, 1e-06 |
| P2/256 | X | 1e-03, 1e-04, 1e-05 |
| P2/256 | sensitivity | 1e-03, 1e-04, 1e-05 |
| P3/64 | E0a | 1e-03, 1e-04 |
| P3/64 | E0b | 1e-03, 1e-04 |
| P3/64 | E1 | 1e-03, 1e-04 |
| P3/64 | S0 | 1e-03 |
| P3/64 | S1a | 1e-03 |
| P3/64 | S1b | 1e-03 |
| P3/64 | S1c | 1e-03, 1e-04 |
| P3/64 | X | 1e-03 |
| P3/64 | sensitivity | 1e-03 |
| P4/64 | E0a | 1e-03, 1e-04, 1e-05 |
| P4/64 | E0b | 1e-03, 1e-04, 1e-05 |
| P4/64 | E1 | 1e-03, 1e-04, 1e-05 |
| P4/64 | S0 | 1e-03, 1e-04 |
| P4/64 | S1a | 1e-03, 1e-04 |
| P4/64 | S1b | 1e-03, 1e-04 |
| P4/64 | S1c | 1e-03, 1e-04 |
| P4/64 | X | 1e-03 |
| P4/64 | sensitivity | 1e-03, 1e-04 |
| P5/128 | E0a | 1e-03, 1e-04, 1e-05 |
| P5/128 | E0b | 1e-03, 1e-04, 1e-05 |
| P5/128 | E1 | 1e-03, 1e-04, 1e-05 |
| P5/128 | S0 | 1e-03, 1e-04 |
| P5/128 | S1a | 1e-03, 1e-04 |
| P5/128 | S1b | 1e-03, 1e-04 |
| P5/128 | S1c | 1e-03, 1e-04, 1e-05 |
| P5/128 | X | none |
| P5/128 | sensitivity | 1e-03, 1e-04 |
| P5/64 | E0a | 1e-03, 1e-04, 1e-05, 1e-06 |
| P5/64 | E0b | 1e-03, 1e-04, 1e-05, 1e-06 |
| P5/64 | E1 | 1e-03, 1e-04, 1e-05, 1e-06 |
| P5/64 | S0 | 1e-03, 1e-04 |
| P5/64 | S1a | 1e-03, 1e-04, 1e-05 |
| P5/64 | S1b | 1e-03, 1e-04 |
| P5/64 | S1c | 1e-03, 1e-04, 1e-05 |
| P5/64 | X | none |
| P5/64 | sensitivity | 1e-03, 1e-04 |
Source: `benchmarks/results/d3_work_precision.json`, `/matched_targets`.

### H3: fixed-step anchors

| Case | Configuration | Anchor | Target | R |
| --- | --- | --- | ---: | ---: |
| P1/64 | E0a | F | 1e-04 | 0.3707 |
| P1/64 | E0a | F | 1e-05 | 0.3837 |
| P1/64 | E0a | F | 1e-06 | 0.3423 |
| P1/64 | S0 | F | 1e-04 | 0.2609 |
| P1/64 | S0 | F | 1e-05 | 0.3335 |
| P1/64 | S0 | F | 1e-06 | 0.4052 |
| P1/64 | S1a | F | 1e-05 | 0.2765 |
| P1/64 | S1a | F | 1e-06 | 0.331 |
| P1/64 | S1b | F | 1e-05 | 0.3071 |
| P1/64 | S1b | F | 1e-06 | 0.3892 |
| P1/64 | S1c | F | 1e-05 | 0.2623 |
| P1/64 | S1c | F | 1e-06 | 0.3526 |
| P2/128 | E0a | F | 1e-04 | 0.7975 |
| P2/128 | E0a | F | 1e-05 | 0.7104 |
| P2/128 | E0b | F | 1e-04 | 1.587 |
| P2/128 | E0b | F | 1e-05 | 1.766 |
| P2/128 | E1 | F | 1e-04 | 1.587 |
| P2/128 | E1 | F | 1e-05 | 1.766 |
| P2/128 | S0 | F | 1e-04 | 0.8307 |
| P2/128 | S0 | F | 1e-05 | 0.9564 |
| P2/128 | S1a | F | 1e-04 | 0.7788 |
| P2/128 | S1a | F | 1e-05 | 0.8914 |
| P2/128 | S1b | F | 1e-04 | 0.8258 |
| P2/128 | S1b | F | 1e-05 | 0.8973 |
| P2/128 | S1c | F | 1e-04 | 0.7426 |
| P2/128 | S1c | F | 1e-05 | 0.8285 |
| P2/256 | E0a | F | 1e-04 | 0.772 |
| P2/256 | E0a | F | 1e-05 | 0.6619 |
| P2/256 | E0b | F | 1e-04 | 1.555 |
| P2/256 | E0b | F | 1e-05 | 1.656 |
| P2/256 | E1 | F | 1e-04 | 1.555 |
| P2/256 | E1 | F | 1e-05 | 1.656 |
| P2/256 | S0 | F | 1e-04 | 0.8033 |
| P2/256 | S0 | F | 1e-05 | 0.894 |
| P2/256 | S1a | F | 1e-04 | 0.7552 |
| P2/256 | S1a | F | 1e-05 | 0.8301 |
| P2/256 | S1b | F | 1e-04 | 0.7583 |
| P2/256 | S1b | F | 1e-05 | 0.8366 |
| P2/256 | S1c | F | 1e-04 | 0.6572 |
| P2/256 | S1c | F | 1e-05 | 0.7606 |
| P3/64 | E0a | F | 1e-03 | 0.2325 |
| P3/64 | E0a | F | 1e-04 | 0.2261 |
| P3/64 | E0b | F | 1e-03 | 0.7567 |
| P3/64 | E0b | F | 1e-04 | 0.8389 |
| P3/64 | E1 | F | 1e-03 | 0.7567 |
| P3/64 | E1 | F | 1e-04 | 0.8389 |
| P3/64 | S0 | F | 1e-03 | 0.3184 |
| P3/64 | S1a | F | 1e-03 | 0.3259 |
| P3/64 | S1b | F | 1e-03 | 0.3188 |
| P3/64 | S1c | F | 1e-03 | 0.3168 |
| P3/64 | S1c | F | 1e-04 | 0.3375 |
| P4/64 | E0a | F | 1e-03 | 0.1479 |
| P4/64 | E0a | F | 1e-04 | 0.1082 |
| P4/64 | E0a | F | 1e-05 | 0.08955 |
| P4/64 | E0b | F | 1e-03 | 0.4277 |
| P4/64 | E0b | F | 1e-04 | 0.3691 |
| P4/64 | E0b | F | 1e-05 | 0.366 |
| P4/64 | E1 | F | 1e-03 | 0.4277 |
| P4/64 | E1 | F | 1e-04 | 0.3691 |
| P4/64 | E1 | F | 1e-05 | 0.366 |
| P4/64 | S0 | F | 1e-03 | 0.3459 |
| P4/64 | S0 | F | 1e-04 | 0.3143 |
| P4/64 | S1a | F | 1e-03 | 0.4375 |
| P4/64 | S1a | F | 1e-04 | 0.3519 |
| P4/64 | S1b | F | 1e-03 | 0.388 |
| P4/64 | S1b | F | 1e-04 | 0.3285 |
| P4/64 | S1c | F | 1e-03 | 0.3649 |
| P4/64 | S1c | F | 1e-04 | 0.3246 |
| P5/128 | E0a | F | 1e-03 | 2.334 |
| P5/128 | E0a | F_g | 1e-04 | 0.7227 |
| P5/128 | E0a | F_g | 1e-05 | 0.604 |
| P5/128 | E0b | F | 1e-03 | 5.059 |
| P5/128 | E0b | F_g | 1e-04 | 2.332 |
| P5/128 | E0b | F_g | 1e-05 | 2.834 |
| P5/128 | E1 | F | 1e-03 | 5.059 |
| P5/128 | E1 | F_g | 1e-04 | 2.332 |
| P5/128 | E1 | F_g | 1e-05 | 2.834 |
| P5/128 | S0 | F | 1e-03 | 1.508 |
| P5/128 | S0 | F_g | 1e-04 | 0.6258 |
| P5/128 | S1a | F | 1e-03 | 1.563 |
| P5/128 | S1a | F_g | 1e-04 | 0.6492 |
| P5/128 | S1b | F | 1e-03 | 1.464 |
| P5/128 | S1b | F_g | 1e-04 | 0.6378 |
| P5/128 | S1c | F | 1e-03 | 1.426 |
| P5/128 | S1c | F_g | 1e-04 | 0.6233 |
| P5/128 | S1c | F_g | 1e-05 | 0.7262 |
| P5/64 | E0a | F | 1e-03 | 2.085 |
| P5/64 | E0a | F_g | 1e-04 | 0.6355 |
| P5/64 | E0a | F_g | 1e-05 | 0.6053 |
| P5/64 | E0b | F | 1e-03 | 3.756 |
| P5/64 | E0b | F_g | 1e-04 | 2.224 |
| P5/64 | E0b | F_g | 1e-05 | 3.37 |
| P5/64 | E1 | F | 1e-03 | 3.756 |
| P5/64 | E1 | F_g | 1e-04 | 2.224 |
| P5/64 | E1 | F_g | 1e-05 | 3.37 |
| P5/64 | S0 | F | 1e-03 | 1.129 |
| P5/64 | S0 | F_g | 1e-04 | 0.6652 |
| P5/64 | S1a | F | 1e-03 | 1.127 |
| P5/64 | S1a | F_g | 1e-04 | 0.6693 |
| P5/64 | S1a | F_g | 1e-05 | 0.8489 |
| P5/64 | S1b | F | 1e-03 | 1.145 |
| P5/64 | S1b | F_g | 1e-04 | 0.6637 |
| P5/64 | S1c | F | 1e-03 | 1.134 |
| P5/64 | S1c | F_g | 1e-04 | 0.6718 |
| P5/64 | S1c | F_g | 1e-05 | 0.8466 |
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H3`.

H3 structural ceilings against each method's own fixed-step use: ERK32ZB 0.75, ERK43ZB 1, step_doubling 0.6667.
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H3_structural_ceilings`.

### Robustness of the primary comparisons

| Configuration | Case | Median lean-NFE Q | Median confirmed-target Q |
| --- | --- | ---: | ---: |
| E0a | P1/64 | 1.393 | 1.151 |
| E0a | P2/128 | 0.9011 | 0.7428 |
| E0a | P2/256 | 0.8986 | 0.7404 |
| E0a | P3/64 | 0.8926 | 0.7304 |
| E0a | P4/64 | 0.47 | 0.386 |
| E0a | P5/128 | 1.651 | 1.351 |
| E0a | P5/64 | 1.713 | 1.401 |
| E0b | P1/64 | n/a | n/a |
| E0b | P2/128 | 1.725 | 1.877 |
| E0b | P2/256 | 1.706 | 1.853 |
| E0b | P3/64 | 2.179 | 2.377 |
| E0b | P4/64 | 1.103 | 1.206 |
| E0b | P5/128 | 3.268 | 3.54 |
| E0b | P5/64 | 3.066 | 3.335 |
| E1 | P1/64 | n/a | n/a |
| E1 | P2/128 | 1.725 | 1.877 |
| E1 | P2/256 | 1.706 | 1.853 |
| E1 | P3/64 | 2.179 | 2.377 |
| E1 | P4/64 | 1.103 | 1.206 |
| E1 | P5/128 | 3.268 | 3.54 |
| E1 | P5/64 | 3.066 | 3.335 |
| S1a | P1/64 | 0.8291 | 0.8291 |
| S1a | P2/128 | 0.9289 | 0.9375 |
| S1a | P2/256 | 0.926 | 0.9354 |
| S1a | P3/64 | n/a | n/a |
| S1a | P4/64 | n/a | n/a |
| S1a | P5/128 | n/a | n/a |
| S1a | P5/64 | n/a | n/a |
| S1b | P1/64 | 0.9208 | 0.9208 |
| S1b | P2/128 | 0.9896 | 0.9942 |
| S1b | P2/256 | 0.9396 | 0.944 |
| S1b | P3/64 | n/a | n/a |
| S1b | P4/64 | n/a | n/a |
| S1b | P5/128 | n/a | n/a |
| S1b | P5/64 | n/a | n/a |
| S1c | P1/64 | 0.7868 | 0.7868 |
| S1c | P2/128 | 0.8894 | 0.8939 |
| S1c | P2/256 | 0.8139 | 0.8181 |
| S1c | P3/64 | n/a | n/a |
| S1c | P4/64 | n/a | n/a |
| S1c | P5/128 | n/a | n/a |
| S1c | P5/64 | n/a | n/a |
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/lean_NFE_Q` and `/hypotheses/confirmed_target_Q`.

| Configuration | Confirmed-target robustness verdict |
| --- | --- |
| E0a | False |
| E0b | True |
| S1a | neutral |
| S1b | neutral |
| S1c | neutral |
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/confirmed_target_verdicts`.

| Crossover configuration | Problem/case | Lowest Q | Highest Q |
| --- | --- | ---: | ---: |
| E0a | P1 | 0.8447 | 1.421 |
| E0a | P5 | 0.9554 | 1.847 |
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/crossovers`.

E1 selected `E0b` with filter `None`. D2 not proposed upstream: False; D1 at most documented opt-in: True.
Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/E1_selection` and `/hypotheses/negative_result_clause`.

## Timing

| Case/config | Median seconds | IQR seconds | Runs |
| --- | ---: | ---: | ---: |
| P1_64_E0a_0.0001 | 0.01343 | 0.00114 | 50 |
| P1_64_E0a_0.0003 | 0.01263 | 0.00123 | 50 |
| P1_64_F_4 | 0.0008066 | 0.00025 | 50 |
| P1_64_F_8 | 0.001261 | 0.000247 | 50 |
| P1_64_S0_0.0003 | 0.01304 | 0.00156 | 50 |
| P1_64_S0_0.001 | 0.01242 | 0.00128 | 50 |
| P2_128_E0a_0.0001 | 0.2477 | 0.0318 | 50 |
| P2_128_E0a_0.0003 | 0.1962 | 0.0211 | 50 |
| P2_128_E0b_0.0001 | 0.1517 | 0.0194 | 50 |
| P2_128_E0b_0.0003 | 0.142 | 0.0121 | 50 |
| P2_128_E1_0.0001 | 0.152 | 0.0106 | 50 |
| P2_128_E1_0.0003 | 0.142 | 0.00892 | 50 |
| P2_128_F_128 | 0.1124 | 0.0124 | 50 |
| P2_128_F_64 | 0.05406 | 0.0065 | 50 |
| P2_128_S0_1e-05 | 0.1985 | 0.0177 | 50 |
| P2_128_S0_3e-05 | 0.1825 | 0.0219 | 50 |
| P2_256_E0a_0.0001 | 0.6888 | 0.0831 | 50 |
| P2_256_E0a_0.0003 | 0.5658 | 0.089 | 50 |
| P2_256_E0b_0.0001 | 0.4379 | 0.042 | 50 |
| P2_256_E0b_0.0003 | 0.4026 | 0.0336 | 50 |
| P2_256_E1_0.0001 | 0.4388 | 0.0591 | 50 |
| P2_256_E1_0.0003 | 0.3921 | 0.0698 | 50 |
| P2_256_F_128 | 0.208 | 0.0334 | 50 |
| P2_256_F_64 | 0.1047 | 0.0227 | 50 |
| P2_256_S0_1e-05 | 0.5701 | 0.0633 | 50 |
| P2_256_S0_3e-05 | 0.5104 | 0.0658 | 50 |
| P3_64_E0a_1e-07 | 2.99 | 0.262 | 50 |
| P3_64_E0a_3e-07 | 2.076 | 0.156 | 50 |
| P3_64_E0b_1e-07 | 0.5516 | 0.0427 | 50 |
| P3_64_E0b_3e-08 | 0.769 | 0.0666 | 50 |
| P3_64_E1_1e-07 | 0.5634 | 0.0478 | 50 |
| P3_64_E1_3e-08 | 0.7631 | 0.0416 | 50 |
| P3_64_F_1024 | 0.2216 | 0.0094 | 50 |
| P3_64_F_2048 | 0.4461 | 0.02 | 50 |
| P4_64_E0a_1e-07 | 1.427 | 0.0226 | 50 |
| P4_64_E0a_3e-07 | 1.012 | 0.018 | 50 |
| P4_64_E0b_1e-07 | 0.3449 | 0.015 | 50 |
| P4_64_E0b_3e-07 | 0.2668 | 0.0124 | 50 |
| P4_64_E1_1e-07 | 0.3444 | 0.00985 | 50 |
| P4_64_E1_3e-07 | 0.2666 | 0.0137 | 50 |
| P4_64_F_256 | 0.05687 | 0.00165 | 50 |
| P4_64_F_512 | 0.1123 | 0.00334 | 50 |
| P4_64_S0_1e-08 | 0.3865 | 0.043 | 50 |
| P4_64_S0_3e-08 | 0.3112 | 0.0294 | 50 |
| P5_128_E0a_1e-05 | 11.35 | 0.0799 | 50 |
| P5_128_E0a_3e-06 | 18.72 | 0.157 | 50 |
| P5_128_E0b_1e-06 | 5.911 | 0.0734 | 50 |
| P5_128_E0b_3e-06 | 4.193 | 0.0589 | 50 |
| P5_128_E1_1e-06 | 5.859 | 0.093 | 50 |
| P5_128_E1_3e-06 | 4.178 | 0.0545 | 50 |
| P5_128_S0_1e-07 | 9.368 | 0.441 | 50 |
| P5_128_S0_3e-08 | 14.08 | 0.703 | 50 |
| P5_64_E0a_1e-05 | 5.365 | 0.771 | 50 |
| P5_64_E0a_3e-06 | 8.747 | 1.34 | 50 |
| P5_64_E0b_1e-06 | 2.541 | 0.414 | 50 |
| P5_64_E0b_3e-06 | 1.766 | 0.299 | 50 |
| P5_64_E1_1e-06 | 2.573 | 0.42 | 50 |
| P5_64_E1_3e-06 | 1.736 | 0.275 | 50 |
| P5_64_S0_1e-07 | 5.585 | 1.85 | 50 |
| P5_64_S0_3e-08 | 8.545 | 2.98 | 50 |
Source: `benchmarks/results/d3_work_precision.json`, `/timing/records`.

| Unbracketed case | Configuration | Reason |
| --- | --- | --- |
| P1/64 | E0b | method ineligible |
| P1/64 | E1 | method ineligible |
| P3/64 | S0 | e*=1e-4 outside valid Pareto error range |
| P5/64 | F | e*=1e-4 outside valid Pareto error range |
| P5/128 | F | e*=1e-4 outside valid Pareto error range |
Source: `benchmarks/results/d3_work_precision.json`, `/gates/stage_b/no_bracket`.

| Case/config | Interpolated seconds at target | Time Q versus S0 | Bootstrap 95% CI |
| --- | ---: | ---: | --- |
| P1/64/E0a | 0.01269 | 0.9835 | [0.946, 1.014] |
| P1/64/F | 0.001251 | 9.98 | [9.286, 10.4] |
| P1/64/S0 | 0.01248 | n/a | n/a |
| P2/128/E0a | 0.2058 | 0.9007 | [0.8643, 0.9376] |
| P2/128/E0b | 0.1438 | 1.289 | [1.247, 1.339] |
| P2/128/E1 | 0.1438 | 1.289 | [1.256, 1.336] |
| P2/128/F | 0.08602 | 2.154 | [2.083, 2.254] |
| P2/128/S0 | 0.1853 | n/a | n/a |
| P2/256/E0a | 0.604 | 0.8734 | [0.8311, 0.9195] |
| P2/256/E0b | 0.4179 | 1.262 | [1.225, 1.321] |
| P2/256/E1 | 0.4121 | 1.28 | [1.234, 1.337] |
| P2/256/F | 0.1627 | 3.243 | [3.077, 3.394] |
| P2/256/S0 | 0.5275 | n/a | n/a |
| P3/64/E0a | 2.207 | n/a | n/a |
| P3/64/E0b | 0.6151 | n/a | n/a |
| P3/64/E1 | 0.6223 | n/a | n/a |
| P3/64/F | 0.2673 | n/a | n/a |
| P4/64/E0a | 1.032 | 0.3056 | [0.3023, 0.3136] |
| P4/64/E0b | 0.3093 | 1.02 | [1.007, 1.046] |
| P4/64/E1 | 0.3089 | 1.021 | [1.008, 1.047] |
| P4/64/F | 0.06015 | 5.242 | [5.177, 5.37] |
| P4/64/S0 | 0.3153 | n/a | n/a |
| P5/128/E0a | 12.31 | 1.091 | [1.081, 1.104] |
| P5/128/E0b | 4.312 | 3.115 | [3.085, 3.155] |
| P5/128/E1 | 4.295 | 3.127 | [3.095, 3.166] |
| P5/128/S0 | 13.43 | n/a | n/a |
| P5/64/E0a | 5.947 | 1.171 | [1.064, 1.225] |
| P5/64/E0b | 1.782 | 3.91 | [3.538, 4.14] |
| P5/64/E1 | 1.752 | 3.976 | [3.606, 4.302] |
| P5/64/S0 | 6.966 | n/a | n/a |
Source: `benchmarks/results/d3_work_precision.json`, `/timing/interpolated` and `/timing/ratios`.

## Estimator-power probes and time dependence

| Case | Method | State fraction | Six slopes | Median k>=2 | Nominal |
| --- | --- | ---: | --- | ---: | ---: |
| P1/64 | S0 | 0.05 | 4.82, 4.91, 4.95, 4.98, 4.99, 4.45 | 4.96 | 5 |
| P1/64 | E0a | 0.05 | 2.96, 2.93, 2.96, 2.98, 2.99, 2.99 | 2.98 | 3 |
| P1/64 | S0 | 0.5 | 4.68, 4.83, 4.91, 4.96, 4.98, 3.41 | 4.94 | 5 |
| P1/64 | E0a | 0.5 | 2.72, 2.86, 2.93, 2.97, 2.98, 2.99 | 2.97 | 3 |
| P1/64 | S0 | 0.95 | 4.35, 4.67, 4.83, 4.91, 4.83, 1.19 | 4.83 | 5 |
| P1/64 | E0a | 0.95 | 3.25, 3.13, 3.07, 3.03, 3.02, 3.01 | 3.03 | 3 |
| P2/128 | S0 | 0.05 | 6.06, 5.58, 5.29, 5.15, 5.07, 5.04 | 5.11 | 5 |
| P2/128 | E0a | 0.05 | 5.17, 1.81, 2.65, 2.88, 2.95, 2.98 | 2.91 | 3 |
| P2/128 | E0b | 0.05 | 4.31, 4.32, 4.19, 4.1, 4.05, 4.03 | 4.08 | 4 |
| P2/128 | S0 | 0.5 | 4.91, 4.96, 4.98, 4.99, 4.89, 0.784 | 4.94 | 5 |
| P2/128 | E0a | 0.5 | 2.93, 2.97, 2.98, 2.99, 3, 3 | 2.99 | 3 |
| P2/128 | E0b | 0.5 | 4.03, 4.01, 4.01, 4, 4, 4 | 4 | 4 |
| P2/128 | S0 | 0.95 | 5.1, 5.05, 5.03, 5.01, 5.01, 4.82 | 5.01 | 5 |
| P2/128 | E0a | 0.95 | 3.05, 3.03, 3.01, 3.01, 3, 3 | 3.01 | 3 |
| P2/128 | E0b | 0.95 | 4.2, 4.1, 4.05, 4.03, 4.01, 4.01 | 4.02 | 4 |
| P2/256 | S0 | 0.05 | 5.91, 5.48, 5.24, 5.12, 5.06, 5.03 | 5.09 | 5 |
| P2/256 | E0a | 0.05 | 3.99, 1.99, 2.74, 2.9, 2.96, 2.98 | 2.93 | 3 |
| P2/256 | E0b | 0.05 | 4.38, 4.29, 4.16, 4.09, 4.04, 4.02 | 4.06 | 4 |
| P2/256 | S0 | 0.5 | 4.95, 4.97, 4.99, 5.01, 4.08, -0.253 | 4.54 | 5 |
| P2/256 | E0a | 0.5 | 2.93, 2.97, 2.98, 2.99, 3, 3 | 2.99 | 3 |
| P2/256 | E0b | 0.5 | 4.02, 4.01, 4.01, 4, 4, 4 | 4 | 4 |
| P2/256 | S0 | 0.95 | 5.08, 5.04, 5.02, 5.01, 5.01, 4.19 | 5.01 | 5 |
| P2/256 | E0a | 0.95 | 3.04, 3.02, 3.01, 3.01, 3, 3 | 3 | 3 |
| P2/256 | E0b | 0.95 | 4.18, 4.09, 4.05, 4.02, 4.01, 4.01 | 4.02 | 4 |
| P3/64 | S0 | 0.05 | 4.58, 4.34, 4.27, 4.46, 4.67, 4.81 | 4.56 | 5 |
| P3/64 | E0a | 0.05 | 2.63, 2.42, 2.54, 2.7, 2.83, 2.91 | 2.77 | 3 |
| P3/64 | E0b | 0.05 | 4.59, 4.21, 3.83, 3.72, 3.79, 3.87 | 3.81 | 4 |
| P3/64 | S0 | 0.5 | 4.23, 3.86, 4.14, 4.5, 4.71, 4.84 | 4.6 | 5 |
| P3/64 | E0a | 0.5 | 2.04, 2.31, 2.59, 2.77, 2.88, 2.94 | 2.82 | 3 |
| P3/64 | E0b | 0.5 | 3.38, 3.76, 3.89, 3.94, 3.97, 3.98 | 3.95 | 4 |
| P3/64 | S0 | 0.95 | 3.64, 3.11, 2.78, 3.78, 4.23, 3.77 | 3.77 | 5 |
| P3/64 | E0a | 0.95 | 1.96, 1.88, 2.05, 2.32, 2.54, 2.68 | 2.43 | 3 |
| P3/64 | E0b | 0.95 | 2.65, 3.52, 3.28, 1.44, 2.39, 3 | 2.69 | 4 |
| P4/64 | S0 | 0.05 | 4.54, 4.76, 4.88, 4.93, 3.55, -0.119 | 4.22 | 5 |
| P4/64 | E0a | 0.05 | 2.73, 2.84, 2.91, 2.95, 2.98, 2.99 | 2.97 | 3 |
| P4/64 | E0b | 0.05 | 3.67, 3.83, 3.91, 3.96, 3.98, 3.97 | 3.96 | 4 |
| P4/64 | S0 | 0.5 | 4.94, 4.97, 4.98, 4.99, 4.99, 3.6 | 4.99 | 5 |
| P4/64 | E0a | 0.5 | 3.06, 3.03, 3.01, 3.01, 3, 3 | 3 | 3 |
| P4/64 | E0b | 0.5 | 4.07, 4.04, 4.02, 4.01, 4.01, 4 | 4.01 | 4 |
| P4/64 | S0 | 0.95 | 4.64, 4.78, 4.88, 4.94, 4.97, 4.98 | 4.95 | 5 |
| P4/64 | E0a | 0.95 | 2.5, 2.75, 2.88, 2.94, 2.97, 2.99 | 2.96 | 3 |
| P4/64 | E0b | 0.95 | 3.92, 3.87, 3.94, 3.97, 3.98, 3.99 | 3.98 | 4 |
| P5/64 | S0 | 0.05 | 3.55, 2.18, 2.78, 3.59, 3.23, 2.86 | 3.05 | 5 |
| P5/64 | E0a | 0.05 | 1.51, 1.57, 1.81, 2.07, 2.28, 2.42 | 2.17 | 3 |
| P5/64 | E0b | 0.05 | 2.79, 2.97, 3.19, 3.15, 3.12, 2.81 | 3.13 | 4 |
| P5/64 | S0 | 0.5 | 3.18, 2.18, 3.04, 3.44, 2.34, 3.42 | 3.23 | 5 |
| P5/64 | E0a | 0.5 | 1.53, 1.57, 1.82, 2.1, 2.32, 2.32 | 2.21 | 3 |
| P5/64 | E0b | 0.5 | 2.94, 1.76, 0.412, 1.4, 1.96, 2.49 | 1.68 | 4 |
| P5/64 | S0 | 0.95 | 3.13, 2.27, 3.21, 2.32, 2.5, 3.68 | 2.86 | 5 |
| P5/64 | E0a | 0.95 | 1.53, 1.58, 1.83, 2.11, 2.27, 2.08 | 2.1 | 3 |
| P5/64 | E0b | 0.95 | 3.84, -0.923, 0.827, 1.45, 1.95, 2.45 | 1.7 | 4 |
| P5/128 | S0 | 0.05 | 3.15, 2.6, 2.7, 3.17, 3.29, 3.01 | 3.09 | 5 |
| P5/128 | E0a | 0.05 | 1.54, 1.53, 1.77, 2.03, 2.26, 2.43 | 2.14 | 3 |
| P5/128 | E0b | 0.05 | 2.77, 2.87, 3.16, 3.28, 2.91, 2.43 | 3.04 | 4 |
| P5/128 | S0 | 0.5 | 3.18, 2.24, 3.03, 3.12, 2.38, 3.52 | 3.07 | 5 |
| P5/128 | E0a | 0.5 | 1.54, 1.56, 1.8, 2.09, 2.31, 2.26 | 2.17 | 3 |
| P5/128 | E0b | 0.5 | 3.24, 0.688, 0.593, 1.39, 1.92, 2.42 | 1.66 | 4 |
| P5/128 | S0 | 0.95 | 1.24, 2.62, 3.62, 3.23, 2.64, 3.13 | 3.18 | 5 |
| P5/128 | E0a | 0.95 | 1.49, 1.64, 1.76, 1.9, 1.9, 1.86 | 1.88 | 3 |
| P5/128 | E0b | 0.95 | 1.94, 1.83, 1.63, 1.93, 2.36, 2.75 | 2.15 | 4 |
Source: `benchmarks/results/d3_work_precision.json`, `/estimator_power`.

| Case | rtol | Attempts equal | Endpoints equal |
| --- | ---: | --- | --- |
| P2/128 | 1e-04 | True | True |
| P2/128 | 1e-06 | True | True |
| P2/256 | 1e-04 | True | True |
| P2/256 | 1e-06 | True | True |
| P3/64 | 1e-04 | True | True |
| P3/64 | 1e-06 | True | True |
| P4/64 | 1e-04 | True | True |
| P4/64 | 1e-06 | True | True |
| P5/64 | 1e-04 | True | True |
| P5/64 | 1e-06 | True | True |
| P5/128 | 1e-04 | True | True |
| P5/128 | 1e-06 | True | True |
Source: `benchmarks/results/d3_work_precision.json`, `/rhs_time_controls`.

## Scope

The JSON carries per-target Q and R against F/F_g, lean-NFE robustness values, confirmed-target values, attainable targets and failure rows. No extrapolation was used.
Source: `benchmarks/results/d3_work_precision.json`, `/matched_targets`, `/hypotheses`, and `/stage_a`.
