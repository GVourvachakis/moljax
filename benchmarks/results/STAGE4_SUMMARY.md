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

The fresh 21-repetition, two-warmup LOO run has mean and worst regret 1.000.
That is an easy-decision result, not evidence that the calibration is a strong
predictor: 0 of 8 backend-comparable held-out sizes are crossover-adjacent
(second-best backend within 1.5×); all 8 are blowouts. In the current run,
CPU is fastest at all four 1D sizes and at one 2D size, while GPU is fastest at
the other three 2D sizes. A workload spanning the CPU/GPU crossover is needed
to test dispatch discrimination. This is the same "perfect regret" shape
Pavlov flagged as overfitting-adjacent: it is a property of the easy ladder,
not proof of a generally predictive dispatcher.

The interpretation block records per-backend q2. In this particular repeat,
they are 0.911 (1D CPU), 0.695 (1D GPU), 0.953 (2D CPU), and 0.986 (2D GPU),
so no current model is flagged negative. This does not rescue the predictor
claim: the immediately preceding matched-protocol JSON
(`benchmarks/results/device_dispatch_cv.json` at commit `9c7a249`) reported
CPU q2 values -15.091 (1D) and -21.890 (2D). That sign change under the same
timing protocol makes fit quality timing-sensitive; perfect regret in the
blowout-only ladder is not a validation of general predictive accuracy.

## Status

This is a proactive experimental contribution, not a PR-ready public API. A
crossover-regime workload is required before the dispatcher validation is
compelling. The scaling reproduction remains useful on its own as an honest
confirmation of the local FFT/GPU size trend.
