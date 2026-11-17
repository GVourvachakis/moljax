# D3b fixed-pair decomposition — exploratory, post hoc

This analysis was formed after seeing D3. It changes no D3 verdict.

## Provenance and checks

- Base `3a341b949e3b90c6b98eee1b2af67c3146538bbd`; Python 3.12.3; JAX 0.11.2; backend cpu.
Source: `d3b_fixed_pairs.json`, `/provenance`.

- As-run NFE per Z43/Z32 step: 5/4; bitwise D3 spot checks: 4/4.
Source: `d3b_fixed_pairs.json`, `/nfe_check` and `/spot_checks`.

## Fixed-step ladders

Z32's FSAL NFE is hypothetical; the as-run stepper makes four calls per step. A dash for order means the adjacent error ratio was unavailable.

| Case | Ladder | Steps | NFE | FSAL NFE | Relative error | Order | Validity/status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| P1/64 | F | 4 | 16 | — | 0.0024856 | — | valid (exact) |
| P1/64 | F | 8 | 32 | — | 9.43467e-05 | 4.72 | valid (exact) |
| P1/64 | F | 16 | 64 | — | 4.97973e-06 | 4.24 | valid (exact) |
| P1/64 | F | 32 | 128 | — | 2.86256e-07 | 4.12 | valid (exact) |
| P1/64 | F | 64 | 256 | — | 1.71438e-08 | 4.06 | valid (exact) |
| P1/64 | F | 128 | 512 | — | 1.04858e-09 | 4.03 | valid (exact) |
| P2/128 | F | 4 | 16 | — | — | — | non-finite |
| P2/128 | F | 8 | 32 | — | — | — | non-finite |
| P2/128 | F | 16 | 64 | — | 0.238833 | — | valid |
| P2/128 | F | 32 | 128 | — | 0.018219 | 3.71 | valid |
| P2/128 | F | 64 | 256 | — | 0.00063118 | 4.85 | valid |
| P2/128 | F | 128 | 512 | — | 3.46333e-05 | 4.19 | valid |
| P2/128 | F | 256 | 1024 | — | 2.08778e-06 | 4.05 | valid |
| P2/256 | F | 8 | 32 | — | — | — | non-finite |
| P2/256 | F | 16 | 64 | — | 0.266926 | — | valid |
| P2/256 | F | 32 | 128 | — | 0.0208915 | 3.68 | valid |
| P2/256 | F | 64 | 256 | — | 0.000684239 | 4.93 | valid |
| P2/256 | F | 128 | 512 | — | 3.41667e-05 | 4.32 | valid |
| P2/256 | F | 256 | 1024 | — | 1.83717e-06 | 4.22 | valid |
| P3/64 | F | 16 | 64 | — | — | — | non-finite |
| P3/64 | F | 32 | 128 | — | — | — | non-finite |
| P3/64 | F | 64 | 256 | — | — | — | non-finite |
| P3/64 | F | 128 | 512 | — | 0.241779 | — | valid |
| P3/64 | F | 256 | 1024 | — | 0.00700749 | 5.11 | valid |
| P3/64 | F | 512 | 2048 | — | 0.0011983 | 2.55 | valid |
| P3/64 | F | 1024 | 4096 | — | 0.000173694 | 2.79 | valid |
| P3/64 | F | 2048 | 8192 | — | 2.2147e-05 | 2.97 | valid, reference-limited |
| P4/64 | F | 16 | 64 | — | — | — | non-finite |
| P4/64 | F | 32 | 128 | — | — | — | non-finite |
| P4/64 | F | 64 | 256 | — | — | — | non-finite |
| P4/64 | F | 128 | 512 | — | 0.00488308 | — | valid |
| P4/64 | F | 256 | 1024 | — | 0.000125873 | 5.28 | valid |
| P4/64 | F | 512 | 2048 | — | 7.76989e-06 | 4.02 | valid |
| P4/64 | F | 1024 | 4096 | — | 5.01464e-07 | 3.95 | valid, reference-limited |
| P5/64 | F | 128 | 512 | — | — | — | non-finite |
| P5/64 | F | 256 | 1024 | — | — | — | non-finite |
| P5/64 | F | 512 | 2048 | — | — | — | non-finite |
| P5/64 | F | 1024 | 4096 | — | 0.246428 | — | valid |
| P5/64 | F | 2048 | 8192 | — | 0.00351787 | 6.13 | valid |
| P5/64 | F | 4096 | 16384 | — | 0.00127135 | 1.47 | valid |
| P5/64 | F | 8192 | 32768 | — | 0.000454575 | 1.48 | valid |
| P5/64 | F | 16384 | 65536 | — | 0.000174941 | 1.38 | valid |
| P5/64 level 0 | F_g | 4608 | 18432 | — | 0.000305944 | — | valid |
| P5/64 level 1 | F_g | 9216 | 36864 | — | 7.6166e-05 | 2.01 | valid |
| P5/64 level 2 | F_g | 18432 | 73728 | — | 1.4755e-05 | 2.37 | valid |
| P5/64 level 3 | F_g | 36864 | 147456 | — | 1.93289e-06 | 2.93 | valid, reference-limited |
| P5/64 level 4 | F_g | 73728 | 294912 | — | 1.8469e-07 | 3.39 | indeterminate |
| P5/128 | F | 128 | 512 | — | — | — | non-finite |
| P5/128 | F | 256 | 1024 | — | — | — | non-finite |
| P5/128 | F | 512 | 2048 | — | — | — | non-finite |
| P5/128 | F | 1024 | 4096 | — | 0.238293 | — | valid |
| P5/128 | F | 2048 | 8192 | — | 0.0153948 | 3.95 | valid |
| P5/128 | F | 4096 | 16384 | — | 0.00233151 | 2.72 | valid |
| P5/128 | F | 8192 | 32768 | — | 0.000618352 | 1.91 | valid |
| P5/128 | F | 16384 | 65536 | — | 0.000315848 | 0.969 | valid |
| P5/128 level 0 | F_g | 4608 | 18432 | — | 0.000305739 | — | valid |
| P5/128 level 1 | F_g | 9216 | 36864 | — | 5.7897e-05 | 2.4 | valid |
| P5/128 level 2 | F_g | 18432 | 73728 | — | 8.47235e-06 | 2.77 | valid |
| P5/128 level 3 | F_g | 36864 | 147456 | — | 8.86404e-07 | 3.26 | valid, reference-limited |
| P5/128 level 4 | F_g | 73728 | 294912 | — | 7.37367e-08 | 3.59 | indeterminate |
| P1/64 | Z32 | 4 | 16 | 13 | 0.00416837 | — | valid (exact) |
| P1/64 | Z32 | 8 | 32 | 25 | 0.000471025 | 3.15 | valid (exact) |
| P1/64 | Z32 | 16 | 64 | 49 | 5.2191e-05 | 3.17 | valid (exact) |
| P1/64 | Z32 | 32 | 128 | 97 | 6.05548e-06 | 3.11 | valid (exact) |
| P1/64 | Z32 | 64 | 256 | 193 | 7.27518e-07 | 3.06 | valid (exact) |
| P1/64 | Z32 | 128 | 512 | 385 | 8.91156e-08 | 3.03 | valid (exact) |
| P2/128 | Z43 | 4 | 20 | — | — | — | non-finite |
| P2/128 | Z43 | 8 | 40 | — | — | — | non-finite |
| P2/128 | Z43 | 16 | 80 | — | 0.0822598 | — | valid |
| P2/128 | Z43 | 32 | 160 | — | 0.00651199 | 3.66 | valid |
| P2/128 | Z43 | 64 | 320 | — | 0.000203783 | 5 | valid |
| P2/128 | Z43 | 128 | 640 | — | 9.9225e-06 | 4.36 | valid |
| P2/128 | Z43 | 256 | 1280 | — | 5.5201e-07 | 4.17 | valid |
| P2/128 | Z32 | 4 | 16 | 13 | — | — | non-finite |
| P2/128 | Z32 | 8 | 32 | 25 | — | — | non-finite |
| P2/128 | Z32 | 16 | 64 | 49 | 0.729392 | — | valid |
| P2/128 | Z32 | 32 | 128 | 97 | 0.00563582 | 7.02 | valid |
| P2/128 | Z32 | 64 | 256 | 193 | 0.000235898 | 4.58 | valid |
| P2/128 | Z32 | 128 | 512 | 385 | 2.19911e-05 | 3.42 | valid |
| P2/128 | Z32 | 256 | 1024 | 769 | 4.36943e-06 | 2.33 | valid |
| P2/128 | Z32 | 512 | 2048 | 1537 | 6.56659e-07 | 2.73 | valid |
| P2/256 | Z43 | 8 | 40 | — | — | — | non-finite |
| P2/256 | Z43 | 16 | 80 | — | 0.0847417 | — | valid |
| P2/256 | Z43 | 32 | 160 | — | 0.00752526 | 3.49 | valid |
| P2/256 | Z43 | 64 | 320 | — | 0.000252291 | 4.9 | valid |
| P2/256 | Z43 | 128 | 640 | — | 1.62593e-05 | 3.96 | valid |
| P2/256 | Z43 | 256 | 1280 | — | 1.23997e-06 | 3.71 | valid |
| P2/256 | Z43 | 512 | 2560 | — | 8.90285e-08 | 3.8 | valid |
| P2/256 | Z32 | 8 | 32 | 25 | — | — | non-finite |
| P2/256 | Z32 | 16 | 64 | 49 | 0.82075 | — | valid |
| P2/256 | Z32 | 32 | 128 | 97 | 0.00630575 | 7.02 | valid |
| P2/256 | Z32 | 64 | 256 | 193 | 0.000300885 | 4.39 | valid |
| P2/256 | Z32 | 128 | 512 | 385 | 2.90529e-05 | 3.37 | valid |
| P2/256 | Z32 | 256 | 1024 | 769 | 3.0186e-06 | 3.27 | valid |
| P2/256 | Z32 | 512 | 2048 | 1537 | 4.65458e-07 | 2.7 | valid |
| P3/64 | Z43 | 16 | 80 | — | — | — | non-finite |
| P3/64 | Z43 | 32 | 160 | — | — | — | non-finite |
| P3/64 | Z43 | 64 | 320 | — | — | — | non-finite |
| P3/64 | Z43 | 128 | 640 | — | — | — | non-finite |
| P3/64 | Z43 | 256 | 1280 | — | 0.00390076 | — | valid |
| P3/64 | Z43 | 512 | 2560 | — | 0.000258929 | 3.91 | valid |
| P3/64 | Z43 | 1024 | 5120 | — | 3.30038e-05 | 2.97 | valid |
| P3/64 | Z43 | 2048 | 10240 | — | 7.89619e-06 | 2.06 | valid, reference-limited |
| P3/64 | Z43 | 4096 | 20480 | — | 1.36719e-06 | 2.53 | indeterminate |
| P3/64 | Z43 | 8192 | 40960 | — | 2.92472e-07 | 2.22 | indeterminate |
| P3/64 | Z32 | 16 | 64 | 49 | — | — | non-finite |
| P3/64 | Z32 | 32 | 128 | 97 | — | — | non-finite |
| P3/64 | Z32 | 64 | 256 | 193 | — | — | non-finite |
| P3/64 | Z32 | 128 | 512 | 385 | 0.032465 | — | valid |
| P3/64 | Z32 | 256 | 1024 | 769 | 0.00460813 | 2.82 | valid |
| P3/64 | Z32 | 512 | 2048 | 1537 | 0.000958902 | 2.26 | valid |
| P3/64 | Z32 | 1024 | 4096 | 3073 | 0.000435711 | 1.14 | valid |
| P3/64 | Z32 | 2048 | 8192 | 6145 | 0.000107007 | 2.03 | valid |
| P3/64 | Z32 | 4096 | 16384 | 12289 | 1.84391e-05 | 2.54 | valid, reference-limited |
| P3/64 | Z32 | 8192 | 32768 | 24577 | 2.63419e-06 | 2.81 | indeterminate |
| P3/64 | Z32 | 16384 | 65536 | 49153 | 3.70205e-07 | 2.83 | indeterminate |
| P4/64 | Z43 | 16 | 80 | — | — | — | non-finite |
| P4/64 | Z43 | 32 | 160 | — | — | — | non-finite |
| P4/64 | Z43 | 64 | 320 | — | — | — | non-finite |
| P4/64 | Z43 | 128 | 640 | — | 0.00129306 | — | valid |
| P4/64 | Z43 | 256 | 1280 | — | 2.66092e-05 | 5.6 | valid |
| P4/64 | Z43 | 512 | 2560 | — | 1.38846e-06 | 4.26 | valid, reference-limited |
| P4/64 | Z43 | 1024 | 5120 | — | 1.04955e-07 | 3.73 | indeterminate |
| P4/64 | Z32 | 16 | 64 | 49 | — | — | non-finite |
| P4/64 | Z32 | 32 | 128 | 97 | — | — | non-finite |
| P4/64 | Z32 | 64 | 256 | 193 | — | — | non-finite |
| P4/64 | Z32 | 128 | 512 | 385 | 0.00498228 | — | valid |
| P4/64 | Z32 | 256 | 1024 | 769 | 0.000591552 | 3.07 | valid |
| P4/64 | Z32 | 512 | 2048 | 1537 | 8.04927e-05 | 2.88 | valid |
| P4/64 | Z32 | 1024 | 4096 | 3073 | 1.06653e-05 | 2.92 | valid |
| P4/64 | Z32 | 2048 | 8192 | 6145 | 1.38265e-06 | 2.95 | valid, reference-limited |
| P4/64 | Z32 | 4096 | 16384 | 12289 | 1.83554e-07 | 2.91 | indeterminate |
| P5/64 | Z43 | 128 | 640 | — | — | — | non-finite |
| P5/64 | Z43 | 256 | 1280 | — | — | — | non-finite |
| P5/64 | Z43 | 512 | 2560 | — | — | — | non-finite |
| P5/64 | Z43 | 1024 | 5120 | — | — | — | non-finite |
| P5/64 | Z43 | 2048 | 10240 | — | 0.00310534 | — | valid |
| P5/64 | Z43 | 4096 | 20480 | — | 0.0010057 | 1.63 | valid |
| P5/64 | Z43 | 8192 | 40960 | — | 0.000385516 | 1.38 | valid |
| P5/64 | Z43 | 16384 | 81920 | — | 0.000161347 | 1.26 | valid |
| P5/64 | Z43 | 32768 | 163840 | — | 6.53878e-05 | 1.3 | valid |
| P5/64 | Z43 | 65536 | 327680 | — | 2.29214e-05 | 1.51 | valid |
| P5/64 | Z43 | 131072 | 655360 | — | 5.41162e-06 | 2.08 | valid |
| P5/64 | Z43 | 262144 | 1310720 | — | 7.24668e-07 | 2.9 | valid, reference-limited |
| P5/64 | Z32 | 128 | 512 | 385 | — | — | non-finite |
| P5/64 | Z32 | 256 | 1024 | 769 | — | — | non-finite |
| P5/64 | Z32 | 512 | 2048 | 1537 | — | — | non-finite |
| P5/64 | Z32 | 1024 | 4096 | 3073 | 0.0206804 | — | valid |
| P5/64 | Z32 | 2048 | 8192 | 6145 | 0.00678857 | 1.61 | valid |
| P5/64 | Z32 | 4096 | 16384 | 12289 | 0.00235028 | 1.53 | valid |
| P5/64 | Z32 | 8192 | 32768 | 24577 | 0.000788866 | 1.57 | valid |
| P5/64 | Z32 | 16384 | 65536 | 49153 | 0.000280611 | 1.49 | valid |
| P5/64 | Z32 | 32768 | 131072 | 98305 | 0.000106098 | 1.4 | valid |
| P5/64 | Z32 | 65536 | 262144 | 196609 | 3.87425e-05 | 1.45 | valid |
| P5/64 | Z32 | 131072 | 524288 | 393217 | 1.14177e-05 | 1.76 | valid |
| P5/64 | Z32 | 262144 | 1048576 | 786433 | 2.32659e-06 | 2.29 | valid, reference-limited |
| P5/64 | Z32 | 524288 | 2097152 | 1572865 | 3.5554e-07 | 2.71 | valid, reference-limited |
| P5/64 level 0 | Z43_g | 4608 | 23040 | — | 1.09952e-05 | — | valid |
| P5/64 level 1 | Z43_g | 9216 | 46080 | — | 2.90985e-06 | 1.92 | valid, reference-limited |
| P5/64 level 2 | Z43_g | 18432 | 92160 | — | 3.38612e-07 | 3.1 | valid, reference-limited |
| P5/64 level 3 | Z43_g | 36864 | 184320 | — | 3.04256e-08 | 3.48 | indeterminate |
| P5/64 level 4 | Z43_g | 73728 | 368640 | — | 1.52115e-08 | 1 | indeterminate |
| P5/128 | Z43 | 128 | 640 | — | — | — | non-finite |
| P5/128 | Z43 | 256 | 1280 | — | — | — | non-finite |
| P5/128 | Z43 | 512 | 2560 | — | — | — | non-finite |
| P5/128 | Z43 | 1024 | 5120 | — | — | — | non-finite |
| P5/128 | Z43 | 2048 | 10240 | — | 0.0124894 | — | valid |
| P5/128 | Z43 | 4096 | 20480 | — | 0.00164051 | 2.93 | valid |
| P5/128 | Z43 | 8192 | 40960 | — | 0.0004913 | 1.74 | valid |
| P5/128 | Z43 | 16384 | 81920 | — | 0.000291385 | 0.754 | valid |
| P5/128 | Z43 | 32768 | 163840 | — | 0.000167126 | 0.802 | valid |
| P5/128 | Z43 | 65536 | 327680 | — | 8.01897e-05 | 1.06 | valid |
| P5/128 | Z43 | 131072 | 655360 | — | 3.27419e-05 | 1.29 | valid |
| P5/128 | Z43 | 262144 | 1310720 | — | 1.06498e-05 | 1.62 | valid |
| P5/128 | Z43 | 524288 | 2621440 | — | 2.3816e-06 | 2.16 | valid, reference-limited |
| P5/128 | Z43 | 1048576 | — | — | — | — | not run (time) |
| P5/128 | Z32 | 128 | 512 | 385 | — | — | non-finite |
| P5/128 | Z32 | 256 | 1024 | 769 | — | — | non-finite |
| P5/128 | Z32 | 512 | 2048 | 1537 | — | — | non-finite |
| P5/128 | Z32 | 1024 | 4096 | 3073 | 0.136071 | — | valid |
| P5/128 | Z32 | 2048 | 8192 | 6145 | 0.0322502 | 2.08 | valid |
| P5/128 | Z32 | 4096 | 16384 | 12289 | 0.00599486 | 2.43 | valid |
| P5/128 | Z32 | 8192 | 32768 | 24577 | 0.00121409 | 2.3 | valid |
| P5/128 | Z32 | 16384 | 65536 | 49153 | 0.000433592 | 1.49 | valid |
| P5/128 | Z32 | 32768 | 131072 | 98305 | 0.000223901 | 0.953 | valid |
| P5/128 | Z32 | 65536 | 262144 | 196609 | 0.000111682 | 1 | valid |
| P5/128 | Z32 | 131072 | 524288 | 393217 | 4.88815e-05 | 1.19 | valid |
| P5/128 | Z32 | 262144 | 1048576 | 786433 | 1.80201e-05 | 1.44 | valid |
| P5/128 | Z32 | 524288 | 2097152 | 1572865 | 5.1115e-06 | 1.82 | valid, reference-limited |
| P5/128 | Z32 | 1048576 | — | — | — | — | not run (time) |
| P5/128 level 0 | Z43_g | 4608 | 23040 | — | 1.05261e-05 | — | valid |
| P5/128 level 1 | Z43_g | 9216 | 46080 | — | 1.46224e-06 | 2.85 | valid, reference-limited |
| P5/128 level 2 | Z43_g | 18432 | 92160 | — | 1.64944e-07 | 3.15 | indeterminate |
| P5/128 level 3 | Z43_g | 36864 | 184320 | — | 5.07233e-08 | 1.7 | indeterminate |
| P5/128 level 4 | Z43_g | 73728 | 368640 | — | 3.99517e-08 | 0.344 | indeterminate |
Source: `d3b_fixed_pairs.json`, `/ladders` (every row).

## Decomposition at matched accuracy

| Case | Pair | Baseline | Target | Adaptivity | Method | Control | Product | D3 Q | Difference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1/64 | E0a | F | 1e-04 | 3.8323 | 0.60617 | 0.6115 | 1.4205 | 1.4205 | -2.2e-16 |
| P1/64 | E0a | F | 1e-05 | 2.9989 | 0.49857 | 0.76959 | 1.1507 | 1.1507 | 2.2e-16 |
| P1/64 | E0a | F | 1e-06 | 2.4679 | 0.40957 | 0.83568 | 0.8447 | 0.8447 | -1.1e-16 |
| P2/128 | E0a | F | 1e-04 | 1.2038 | 1.2083 | 0.66004 | 0.96012 | 0.96012 | 0 |
| P2/128 | E0a | F | 1e-05 | 1.0455 | 0.96904 | 0.73312 | 0.74277 | 0.74277 | -1.1e-16 |
| P2/128 | E0b | F | 1e-04 | 1.2038 | 1.055 | 1.5047 | 1.911 | 1.911 | -2.2e-16 |
| P2/128 | E0b | F | 1e-05 | 1.0455 | 1.0889 | 1.6217 | 1.8464 | 1.8464 | 0 |
| P2/256 | E0a | F | 1e-04 | 1.2449 | 1.1254 | 0.68594 | 0.96101 | 0.96101 | -1.1e-16 |
| P2/256 | E0a | F | 1e-05 | 1.1185 | 0.96548 | 0.68561 | 0.74041 | 0.74041 | 1.1e-16 |
| P2/256 | E0b | F | 1e-04 | 1.2449 | 0.98777 | 1.5745 | 1.9361 | 1.9361 | 0 |
| P2/256 | E0b | F | 1e-05 | 1.1185 | 0.93922 | 1.7631 | 1.8522 | 1.8522 | 2.2e-16 |
| P3/64 | E0a | F | 1e-03 | 3.1411 | 1.087 | 0.21392 | 0.73045 | 0.73045 | 0 |
| P3/64 | E0b | F | 1e-03 | 3.1411 | 1.2057 | 0.6276 | 2.3769 | 2.3769 | 0 |
| P4/64 | E0a | F | 1e-03 | 2.8909 | 0.80099 | 0.18462 | 0.4275 | 0.4275 | -5.6e-17 |
| P4/64 | E0a | F | 1e-04 | 3.182 | 0.57094 | 0.18958 | 0.34441 | 0.34441 | 0 |
| P4/64 | E0b | F | 1e-03 | 2.8909 | 1.0319 | 0.41449 | 1.2365 | 1.2365 | 0 |
| P4/64 | E0b | F | 1e-04 | 3.182 | 1.073 | 0.34401 | 1.1745 | 1.1745 | -2.2e-16 |
| P5/64 | E0a | F | 1e-03 | 0.88608 | 0.68334 | 3.051 | 1.8474 | 1.8474 | -2.2e-16 |
| P5/64 | E0b | F | 1e-03 | 0.88608 | 0.93665 | 4.0096 | 3.3277 | 3.3277 | 0 |
| P5/128 | E0a | F | 1e-03 | 0.66319 | 0.68274 | 3.4189 | 1.548 | 1.548 | 0 |
| P5/128 | E0b | F | 1e-03 | 0.66319 | 0.93649 | 5.4022 | 3.3551 | 3.3551 | -4.4e-16 |
Source: `d3b_fixed_pairs.json`, `/decomposition` (every row).

## Graded two-factor view on P5 (exploratory, post hoc)

On P5 the uniform-baseline factors above mix two effects. With uniform steps both F and Z43 are limited by the noisy initial layer (observed orders 0.75–1.4 on the finest shared rungs), and their same-step error ratio is only 1.1–1.4, so the uniform "control" factor also carries the grading of that layer. With the γ = 4 graded ladders as the baseline, R(E0b vs F_g) = [F_g/Z43_g] × [Z43_g/E0b]:
Source: `d3b_graded_decomposition.json`, `/ladders`.

| Case | Target | Method F_g/Z43_g | Control Z43_g/E0b | Product | Target status |
| --- | ---: | ---: | ---: | ---: | --- |
| P5/64 | 1e-05 | 3.5 | 0.97 | 3.4 | self-convergence only |
| P5/128 | 1e-05 | 3.0 | 0.96 | 2.8 | confirmed |
Source: `d3b_graded_decomposition.json`, `/decomposition`.

On the fully valid graded levels the same-mesh error ratio F_g/Z43_g is 28–29×; including reference-limited levels gives 26–44×. Reading: on P5, adaptive ERK43ZB's advantage over the graded ETDRK4 oracle comes from the method. Once the layer is graded, fixed-step ERK43ZB is about 3.0–3.5× cheaper than ETDRK4 at 1e-5, and adaptive ERK43ZB is about as efficient as its own graded oracle. Why ETDRK4 is weaker here is not established. This view changes no D3 verdict.
Source: `d3b_graded_decomposition.json`, `/ladders` and `/decomposition`.

## Missing matched-target cells

| Case | Pair | Baseline | Target | Unavailable factor inputs and reason |
| --- | --- | --- | ---: | --- |
| P1/64 | E0a | F | 1e-03 | S0: outside usable Pareto error range [2.5525e-08, 0.000109407]; E0a: outside usable Pareto error range [3.00553e-09, 0.00030879] |
| P2/128 | E0a | F | 1e-03 | F: outside usable Pareto error range [2.08778e-06, 0.00063118]; E0a: outside usable Pareto error range [6.1286e-09, 0.000439816] |
| P2/128 | E0a | F | 1e-06 | F: outside usable Pareto error range [2.08778e-06, 0.00063118] |
| P2/128 | E0b | F | 1e-03 | F: outside usable Pareto error range [2.08778e-06, 0.00063118]; E0b: outside usable Pareto error range [6.97166e-09, 0.000333844] |
| P2/128 | E0b | F | 1e-06 | F: outside usable Pareto error range [2.08778e-06, 0.00063118] |
| P2/256 | E0a | F | 1e-03 | F: outside usable Pareto error range [1.83717e-06, 0.000684239]; E0a: outside usable Pareto error range [7.14831e-09, 0.00055768] |
| P2/256 | E0a | F | 1e-06 | F: outside usable Pareto error range [1.83717e-06, 0.000684239] |
| P2/256 | E0b | F | 1e-03 | F: outside usable Pareto error range [1.83717e-06, 0.000684239]; E0b: outside usable Pareto error range [7.72314e-09, 0.000448744] |
| P2/256 | E0b | F | 1e-06 | F: outside usable Pareto error range [1.83717e-06, 0.000684239] |
| P3/64 | E0a | F | 1e-04 | S0: outside usable Pareto error range [0.000276335, 0.00636004] |
| P3/64 | E0a | F | 1e-05 | S0: outside usable Pareto error range [0.000276335, 0.00636004]; F: outside usable Pareto error range [2.2147e-05, 0.00700749]; Z32: outside usable Pareto error range [1.84391e-05, 0.00460813]; E0a: outside usable Pareto error range [1.00907e-05, 0.00370943] |
| P3/64 | E0a | F | 1e-06 | S0: outside usable Pareto error range [0.000276335, 0.00636004]; F: outside usable Pareto error range [2.2147e-05, 0.00700749]; Z32: outside usable Pareto error range [1.84391e-05, 0.00460813]; E0a: outside usable Pareto error range [1.00907e-05, 0.00370943] |
| P3/64 | E0b | F | 1e-04 | S0: outside usable Pareto error range [0.000276335, 0.00636004] |
| P3/64 | E0b | F | 1e-05 | S0: outside usable Pareto error range [0.000276335, 0.00636004]; F: outside usable Pareto error range [2.2147e-05, 0.00700749]; E0b: outside usable Pareto error range [1.67519e-05, 0.00766374] |
| P3/64 | E0b | F | 1e-06 | S0: outside usable Pareto error range [0.000276335, 0.00636004]; F: outside usable Pareto error range [2.2147e-05, 0.00700749]; Z43: outside usable Pareto error range [7.89619e-06, 0.00390076]; E0b: outside usable Pareto error range [1.67519e-05, 0.00766374] |
| P4/64 | E0a | F | 1e-05 | S0: outside usable Pareto error range [4.5983e-05, 0.0097921] |
| P4/64 | E0a | F | 1e-06 | S0: outside usable Pareto error range [4.5983e-05, 0.0097921]; Z32: outside usable Pareto error range [1.38265e-06, 0.00498228]; E0a: outside usable Pareto error range [3.46131e-06, 0.00363559] |
| P4/64 | E0b | F | 1e-05 | S0: outside usable Pareto error range [4.5983e-05, 0.0097921] |
| P4/64 | E0b | F | 1e-06 | S0: outside usable Pareto error range [4.5983e-05, 0.0097921]; Z43: outside usable Pareto error range [1.38846e-06, 0.00129306]; E0b: outside usable Pareto error range [6.41423e-06, 0.00607373] |
| P5/64 | E0a | F | 1e-04 | F: outside usable Pareto error range [0.000174941, 0.00351787] |
| P5/64 | E0a | F | 1e-05 | S0: outside usable Pareto error range [2.0225e-05, 0.00663438]; F: outside usable Pareto error range [0.000174941, 0.00351787] |
| P5/64 | E0a | F | 1e-06 | S0: outside usable Pareto error range [2.0225e-05, 0.00663438]; F: outside usable Pareto error range [0.000174941, 0.00351787] |
| P5/64 | E0b | F | 1e-04 | F: outside usable Pareto error range [0.000174941, 0.00351787] |
| P5/64 | E0b | F | 1e-05 | S0: outside usable Pareto error range [2.0225e-05, 0.00663438]; F: outside usable Pareto error range [0.000174941, 0.00351787] |
| P5/64 | E0b | F | 1e-06 | S0: outside usable Pareto error range [2.0225e-05, 0.00663438]; F: outside usable Pareto error range [0.000174941, 0.00351787] |
| P5/64 | E0b | F_g | 1e-03 | F_g: outside usable Pareto error range [1.93289e-06, 0.000305944]; Z43_g: outside usable Pareto error range [3.38612e-07, 1.09952e-05] |
| P5/64 | E0b | F_g | 1e-04 | Z43_g: outside usable Pareto error range [3.38612e-07, 1.09952e-05] |
| P5/64 | E0b | F_g | 1e-05 | S0: outside usable Pareto error range [2.0225e-05, 0.00663438] |
| P5/64 | E0b | F_g | 1e-06 | S0: outside usable Pareto error range [2.0225e-05, 0.00663438]; F_g: outside usable Pareto error range [1.93289e-06, 0.000305944] |
| P5/128 | E0a | F | 1e-04 | F: outside usable Pareto error range [0.000315848, 0.00233151] |
| P5/128 | E0a | F | 1e-05 | S0: outside usable Pareto error range [3.06598e-05, 0.00450607]; F: outside usable Pareto error range [0.000315848, 0.00233151] |
| P5/128 | E0a | F | 1e-06 | S0: outside usable Pareto error range [3.06598e-05, 0.00450607]; F: outside usable Pareto error range [0.000315848, 0.00233151]; Z32: outside usable Pareto error range [5.1115e-06, 0.00599486]; E0a: outside usable Pareto error range [1.52976e-06, 0.00301101] |
| P5/128 | E0b | F | 1e-04 | F: outside usable Pareto error range [0.000315848, 0.00233151] |
| P5/128 | E0b | F | 1e-05 | S0: outside usable Pareto error range [3.06598e-05, 0.00450607]; F: outside usable Pareto error range [0.000315848, 0.00233151] |
| P5/128 | E0b | F | 1e-06 | S0: outside usable Pareto error range [3.06598e-05, 0.00450607]; F: outside usable Pareto error range [0.000315848, 0.00233151]; Z43: outside usable Pareto error range [2.3816e-06, 0.00164051]; E0b: outside usable Pareto error range [1.43514e-06, 0.00440873] |
| P5/128 | E0b | F_g | 1e-03 | F_g: outside usable Pareto error range [8.86404e-07, 0.000305739]; Z43_g: outside usable Pareto error range [1.46224e-06, 1.05261e-05] |
| P5/128 | E0b | F_g | 1e-04 | Z43_g: outside usable Pareto error range [1.46224e-06, 1.05261e-05] |
| P5/128 | E0b | F_g | 1e-05 | S0: outside usable Pareto error range [3.06598e-05, 0.00450607] |
| P5/128 | E0b | F_g | 1e-06 | S0: outside usable Pareto error range [3.06598e-05, 0.00450607]; Z43_g: outside usable Pareto error range [1.46224e-06, 1.05261e-05]; E0b: outside usable Pareto error range [1.43514e-06, 0.00440873] |
Source: `d3b_fixed_pairs.json`, `/missing` (every row).
