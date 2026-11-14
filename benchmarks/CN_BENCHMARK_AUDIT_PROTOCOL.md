# Crank–Nicolson benchmark convergence audit protocol

Pre-registered before measurement on `upstream/main` at `d8d4432`. Run on CPU in
float64. Preserve the three paper benchmark scripts and their committed results.
Store every generated state and run log under
`~/moljax-artifacts/cn-benchmark-audit/`, record SHA-256 hashes, and commit no
binary states. Record the SHA-256 of each benchmark script at this base. CPU
wall times are operational logs only, never performance comparisons.

## M0: exact static error matrices

At each script's paper grid and CN timestep, form for every pseudo-spectral
Fourier mode

`G_k = diag(p_u, p_v) (dt/2) J_R(u*, v*)`, where
`p_s = 1/(1 + (dt/2) D_s |k|²)`.

Use each script's wavenumbers and its homogeneous steady state: Schnakenberg
`(a+b, b/(a+b)²)`, Brusselator `(a, b/a)`, and Gray–Scott's trivial state
`(1, 0)`. Report `rho(G_0)`, `max_k rho(G_k)`, and the count of modes with
`rho(G_k) >= 1`. Repeat with the mean-field coupled inverse at the homogeneous
state; its error matrix must be zero.

## M1 and M2: instrumented trajectories

For all three scripts, use the paper's initial condition and seed, grid,
timestep, final time, tolerance, maximum iteration count, and float64. M1
replicates the written CN loop exactly: initialise the new state at the old
state; at each iteration compute the CN residual, apply the diffusion-only FFT
inverse to its negative, update the state, and store the **lagged**, pre-update
residual used in the loop condition. No line search or Newton correction is
introduced. Record per step the iteration count, exit reason (tolerance, cap,
or non-finite), lagged residual, post-update residual in the benchmark norm
`max|res_u| + max|res_v|`, and finiteness. The cap is 15 for Schnakenberg and
Brusselator, 10 for Gray–Scott.

M2 changes only the inverse: at the start of each step compute the spatial
mean `Jbar_R` of the reaction Jacobian and solve each Fourier-mode 2×2 block
of `I - (dt/2)(diag(D_u,D_v) L_k + Jbar_R)`. Record the minimum block singular
value per step in addition to the M1 fields. This is a diagnostic probe
implemented in the audit script, not a shipped preconditioner. The analogous
probe on the dt-sweep branch is not imported.

Classify **each loop separately**:

- `SOLVES`: every step exits by tolerance, has post-update residual no larger
  than the script tolerance, and all values are finite.
- `DOES NOT SOLVE`: any non-finite value, or more than 1% of steps reach the
  cap with post-update residual above `1e-6`.
- `MARGINAL`: every other case.

Report iteration min, median, mean, and max against the paper's Table 14
footnote of approximately three nonlinear evaluations per CN step. Identify
the first non-finite step, if any.

## M3: Schnakenberg linear systems

Choose four states **only from a loop classified SOLVES**, preferring M1 and
then M2: states after steps 1, 10, 100, and the final step. If neither loop
solves, use the benchmark's ETDRK4 formula at `dt = 5e-4` through `t = 1`,
selecting `t = 0.01, 0.1, 1` plus the initial state for a four-state set;
label all four `transient-only`. Never take a state from an unconverged CN loop.

At each state set `u_new = u_old = state`, `b = -res(state)`, and
`J = I - (dt/2)(D L + J_R(state))`. With unrestarted
`moljax.experimental.pme_conditioning._counted_gmres`, tolerance `1e-8` and
cap 100, solve:

- Left: `P^-1 J x = P^-1 b`, whose measured residual is left-preconditioned.
- Right: `J P^-1 y = b`, whose measured residual is physical.

Do both for diffusion FFT and mean-field coupled inverses; run identity once
per state. Report iterations, convergence and final relative residual. Compute
`kappa_2(P)` exactly from all Fourier-mode 2×2 singular values, and estimate
`rho(I - P^-1 J)` with `k = 50` Arnoldi steps. Record whether the library or a
local Arnoldi implementation was used.

## Fidelity and execution gates

Print each original CN loop alongside its replica and explain every
instrumentation-only difference. On each model, five uninstrumented replica
steps must be bitwise equal to five instrumented steps. Test the M0 formula
against dense 2×2 eigenvalues, the mean-field zero check, and 32×32 fidelity.
Any subprocess test must pass
`env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))`.
Run the new tests in both the CI-mirror Python 3.12 environment with
`PYTHONPATH` unset and the local `.venv`; run Ruff 0.16.5, Black and
`git diff --check`. Commit protocol, implementation and tests before M0–M3.
Stop and report if any individual measurement run exceeds two hours.
