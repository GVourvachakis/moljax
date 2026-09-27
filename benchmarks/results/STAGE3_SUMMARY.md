# Stage 3 experimental summary

## Thesis

This experimental study applies the visited-state field-of-values and pseudospectra decision
procedure from Stage 1 to moljax's shipped periodic Brusselator factory,
`create_brusselator_periodic_fft`. It evaluates the preconditioned linear systems on states that
the backward-Euler/JFNK solve actually visits, rather than on an analytic steady state.

## Setup

The study reuses the shipped periodic FFT diffusion preconditioner, the Stage-1 conditioning
toolbox, and the Stage-2 counted matrix-free GMRES measurement. The fixed-step 256 by 256 run has
a 133128-component two-field linearization. Across its four fixed-dt 256 by 256 states, the
FFT-preconditioned adjoint-identity errors are 2.38e-18, 2.40e-18, 3.50e-18, and 5.82e-18,
while the identity-baseline errors are 5.55e-17, 2.78e-17, 2.36e-16, and 2.06e-16. Thus all
four fixed-dt 256 by 256 states pass the 1e-08 adjoint gate.

Source: `benchmarks/results/brusselator_conditioning_fixed_dt.json`,
`fixed_dt_transition.by_regime`.

## Fixed-step 256 by 256 result

At a fixed backward-Euler step size of 0.2, early and later samples use the same periodic grid,
preconditioner, and time-discretization family. The state-dependent Jacobian changes between
visited states by design; no comparison changes the step size. The table reports the
FFT-preconditioned records, with the matching identity GMRES count shown for cost context.

| regime | time | FFT verdict | origin enclosed | disk rate | FOV imaginary extent | FFT / identity GMRES iterations |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Hopf, early | 0.2 | adequate | no | 0.7701 | 0.7614 | 8 / 88 |
| Hopf, later | 10.0 | adequate | no | 0.7288 | 0.7217 | 8 / 52 |
| Turing, early | 0.2 | adequate | no | 0.4165 | 0.4142 | 8 / 101* |
| Turing, developed | 200.0 | indeterminate | yes | 1.5192 | 1.4829 | 9 / 84 |

Source: `benchmarks/results/brusselator_conditioning_fixed_dt.json`,
`fixed_dt_transition.by_regime`. *The early Turing identity solve did not meet the configured
relative-residual tolerance within its recorded budget; 101 is the recorded count.*

The definitive transition is the Turing trajectory: its FFT-preconditioned verdict changes from
`adequate` at t=0.2 to `indeterminate` at t=200. The developed-state departure is
max|u-a| = 2.2331 and max|v-b/a| = 0.9063. The preconditioned solve still converges in nine
counted GMRES iterations, versus 84 for identity. This is an abstention, not a claim that the
solve failed: once the origin is enclosed by the numerical range, the disk rate 1.5192 is not a
valid convergence factor and the geometric guarantee is void. The un-preconditioned reaction
term therefore makes the procedure return `indeterminate` while the linear solve remains cheap.
This is the same origin-enclosure mechanism observed in the Stage-2 strong-reaction study.

The stored outcome is `fft_adequate_to_indeterminate_in_one_regime_at_fixed_dt`.

## Hopf continuation scope

The original fixed-dt=0.2 Hopf path was `adequate` through t=10, but its continuation ceased
converging at t=18.6. Near the limit cycle, the reaction-Jacobian scale gives dt*rho about 3.6 at
the peaks; this is a Newton-basin loss, not a backward-Euler linear-stability failure. A separate
fixed-dt=0.05 Hopf continuation now reaches t=20 at 256 by 256, with the same timestep for its
early and developed samples. The developed state has max|u-a| = 3.3553 and max|v-b/a| = 2.5014;
under FFT diffusion preconditioning it remains `adequate` (origin outside, disk rate 0.5113) and
the counted GMRES work is 13 iterations. The smaller fixed timestep therefore reaches beyond the
previous t=18.6 stop without changing the Hopf verdict over the recorded t=0.2 to t=20 extent.

This still does not claim that Hopf remains adequate at a long-time attractor; it reports
adequacy only through the completed smaller-timestep continuation.

Sources: `benchmarks/results/brusselator_conditioning_fixed_dt.json`,
`fixed_dt_transition.by_regime.hopf` and `scope`; and
`benchmarks/results/brusselator_conditioning_hopf_continuation.json`,
`fixed_dt_transition.by_regime.hopf` and `scope`.

## Limitation and open problem

On the developed Turing reaction--diffusion state, a regime of practical interest, the decision
procedure abstains: it returns `indeterminate` when the shipped FFT-diffusion preconditioner leaves
the reaction Jacobian un-preconditioned and the origin enters the numerical range. This is a
limitation of the enclosing-disk criterion, not evidence that the linear solve failed: the geometric
guarantee is void precisely where the reaction contribution is important, even if the measured
GMRES solve still converges. The same origin-enclosure mechanism appears in the strong-reaction
experiment in #2.

The open problem is whether the reaction term can be folded into the preconditioner, or whether an
origin-enclosed numerical range needs a second decision criterion that does not rely on the
enclosing disk.

## 64 by 64 exploration

The 64 by 64 screen at dt=0.1 found both early trajectories `adequate` under FFT
preconditioning. Its median Hopf imaginary extent was 0.4083, larger than the Turing value
0.2191. The later 64 by 64 exploration at dt=1 recorded origin enclosure and `indeterminate`
verdicts for all three sampled states in both regimes; its Hopf imaginary extent grew from
3.8577 at t=1 to 4.6661 at t=20.

These screens use different time-discretized operators and are exploratory only. They motivate,
but do not replace, the fixed-step 256 by 256 comparison above, which holds dt fixed.

Sources: `benchmarks/results/brusselator_conditioning.json`, `hopf_vs_turing`; and
`benchmarks/results/brusselator_conditioning_developed.json`, `regime_comparison` and
`hopf_vs_turing`.

## Scope and caveats

- This is a conditioning study; it has no exact-solution max-norm error measurement.
- The fixed-step Turing trajectory reaches t=200 at 256 by 256 resolution. The separate
  fixed-dt=0.05 Hopf continuation reaches t=20, not a full long-time-attractor result.
- The FOV imaginary extent distinguishes oscillatory character in the early screen, but the
  decision verdict is governed by whether the origin is enclosed.
- The study stages experimental evidence only. It does not promote a public API or claim that
  FFT diffusion preconditioning resolves all reaction-driven stiffness.

Sources: `benchmarks/results/brusselator_conditioning_fixed_dt.json`,
`benchmarks/results/brusselator_conditioning.json`, and
`benchmarks/results/brusselator_conditioning_developed.json`.
