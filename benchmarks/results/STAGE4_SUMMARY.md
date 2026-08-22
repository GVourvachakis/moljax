# Stage 4: resolution-aware device dispatch (experimental)

## What was built

`moljax.experimental.device_dispatch` is a staged, pluggable cost-model and
backend-selection utility. The shipped validation uses a two-field FFT
Helmholtz workload, while callers may supply a different cost model for a
different primitive. It is experimental rather than public core API.

## Scaling reproduction

Source: `benchmarks/results/scaling_reproduction.json`.

On this RTX 3090 Ti host, GPU work-normalized time decreases monotonically
across the 64²--1024² ladder: 4.12e-9, 1.08e-9, 3.53e-10, 2.60e-10, and
2.30e-10 seconds / (N² log N). The measured crossover is 128². The GPU
speedup ratios are 0.48×, 2.00×, 24.06×, 22.91×, and 13.25×, respectively.

The ratio is not a monotonic scaling result because the CPU baseline is noisy:
its IQR/median is 32.8% at 256², 64.6% at 512², and 15.8% at 1024². The
reliable observation is the GPU's decreasing normalized cost; the absolute
speedups are hardware-specific. The local RTX 3090 Ti measurements are
therefore compared only for trend with the RTX 5060 reference configuration,
not as portable speedup claims.

## Leave-one-size-out dispatch result

Source: `benchmarks/results/device_dispatch_cv.json`.

The dense 2D ladder samples 48²--128² in steps of 8, plus 160² and 256²,
using 41 repetitions and two excluded warmups. It locates a resolvable part of
the crossover: CPU is robustly faster at 88² (CPU/GPU = 0.792×) and GPU is
robustly faster at 112² (1.359×) and 128² (1.331×). Four of the seven
crossover-adjacent sizes are nevertheless noise-dominated by overlapping
median-plus-IQR bands (48², 96², 104², and 120²).

LOO regret over all 13 held-out sizes is mean 1.075 and worst 1.544. The
meaningful reliability-filtered subset contains the three robust
crossover-adjacent cases: its mean regret is 1.088 and worst regret is 1.263.
The policy chooses the correct backend at 112² and 128² but incorrectly picks
GPU at the robust 88² CPU-win (1.263 regret). Thus this denser ladder rules out
the earlier easy-regime interpretation, but it does not validate the current
dispatcher as a reliable crossover predictor on this hardware: only three
near-crossover cases are resolvable, one is a genuine held-out miss, and four
are noise-dominated.

The held-out fit diagnostics reinforce that limitation: CPU q2 is 0.368 with
L2 error 5.09e-3 seconds, while GPU q2 is -1.222 with L2 error 1.89e-4
seconds. The negative GPU q2 explicitly flags the FFT-primitive model as an
unreliable predictor for this workload/backend. A refined cost model and a
lower-variance crossover measurement are required before promoting dispatch
selection beyond experimental status.

## Status

This is a proactive experimental contribution, not a PR-ready public API. The
dense crossover measurement gives an honest partial test, not a successful
validation of the current predictor. The scaling reproduction remains useful
on its own as an honest confirmation of the local FFT/GPU size trend.
