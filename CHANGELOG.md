# Changelog

All notable changes to moljax are documented here.

## [1.1.1] - 2026-08-03

### Fixed

- **`etd_integrate` now compiles its time-stepping loop.** It previously
  stepped through an eager Python `for` loop, so every step paid full
  XLA dispatch. Long integrations were pathologically slow: a 3,276-step
  ETDRK4 run took 41 s, and the NILT-bridge comparison tests built on it
  ran for many minutes to hours.

  Stepping now runs inside `lax.fori_loop` when only the endpoint is
  retained and `lax.scan` when intermediate states are saved. The method
  dispatch is hoisted out of the loop; ETD2's first step is still taken
  eagerly because it seeds the `N_prev` term from `None`.

  Output is bit-identical to the previous implementation, verified across
  all three methods and seven `save_every` / step-count combinations.
  `tests/test_fft_nilt_bridge.py` went from hanging past 900 s to passing
  in 87 s.

- `compare_nilt_vs_timestepping` requested every intermediate state and
  then used only the last one, which for long horizons on fine grids
  allocated hundreds of MB it immediately discarded. It now retains only
  the endpoint, and calls `block_until_ready` so its reported timings
  are not measuring asynchronous dispatch.

- **Table 3 had archived data but no script to regenerate it.** No file
  in the repository produced the four-column CuPy / nvmath / JAX
  comparison; `benchmark_cupy_fft.py` covers only CuPy versus JAX. Adds
  `benchmarks/benchmark_fft_3lib.py`, which regenerates
  `results/fft_3lib_comparison.json`. CuPy and nvmath are optional; a
  missing library yields a null column instead of aborting.

  On the current stack (nvmath 1.0.0, JAX 0.9.1 rather than the paper's
  0.8/0.8.2) the regenerated values track the published ones at 256²,
  512² and 1024². They differ at 64² and 128², where the measurement is
  launch-latency dominated and the paper reports interquartile ranges as
  large as the medians. The table's conclusion is unaffected: JAX inside
  a compiled loop is competitive with CuPy, so the speedups are
  algorithmic rather than kernel-level.

### Changed

- The two NILT-bridge tests marked `slow` in 1.1.0 are no longer slow and
  run by default again. `pytest` no longer deselects `slow` tests; the
  marker remains registered for the pre-existing one in
  `test_option_a_vs_b.py`, which also runs by default again.

### Known issues

- `compare_nilt_vs_timestepping` reports `tss_steps` from
  `ceil(t_end / dt)` while `etd_integrate` floors internally, so it
  integrates to `floor(t_end/dt) * dt` rather than exactly `t_end`. This
  is pre-existing and accounts for the residual ~9e-5 relative error in
  the time-stepping leg of an otherwise exact linear propagation.

## [1.1.0] - 2026-08-03

Reproducibility release. Brings the public repository in line with the
published paper (*Computer Physics Communications* **326** (2026) 110205,
[doi:10.1016/j.cpc.2026.110205](https://doi.org/10.1016/j.cpc.2026.110205))
and corrects a boundary-condition discrepancy between the paper and the
code.

Release `v1.0.0` (commit `25cd9a3`) remains the archival artifact cited by
the paper and is unchanged.

### Fixed

- **Neumann boundary conditions now use the node-centred DCT-I described
  in Section 3.1.1 of the paper.** The code previously implemented the
  cell-centred DCT-II symbol, `-4/dx² sin²(πk/(2N))`, while the paper
  specified DCT-I. This was also internally inconsistent: the Dirichlet
  path uses the node-centred DST-I symbol, so a mixed Dirichlet/Neumann
  problem combined two different grid layouts.

  `BCType.NEUMANN` now selects the node-centred DCT-I form,
  `-4/dx² sin²(πk/(2(N-1)))`, whose eigenvectors exactly diagonalize the
  `[-2, 2]/dx²` end-row stencil. The previous behaviour is available as
  `BCType.NEUMANN_CELL`, or via `centering='cell'` on the affected
  functions.

  Thanks to Georgios Vakis (Vourvachakis), IACM/IESL-FORTH and University
  of Crete, who identified this while building a coupled two-temperature
  solver on moljax.

- Package version reported `0.1.0` despite the paper citing release
  v1.0.0. Now `1.1.0` in both `pyproject.toml` and `moljax.__version__`.

- README described the paper as submitted to the *Journal of
  Computational Physics* under a previous title. Corrected to the
  published CPC reference with DOI.

- `REPRODUCE.md` listed JAX 0.4.35 / CUDA 12.x / Ubuntu 22.04, which
  matched neither the paper nor the environment the benchmarks ran in.

### Added

- **Node-centred DCT-I transforms**: `dct_I`, `idct_I`, `dct_I_2d`,
  `idct_I_2d`. JAX exposes only the type-2 cosine transform, so these are
  built from the real FFT of the symmetric even extension of length
  `2N-2`. Verified against `scipy.fft.dct(type=1)` to machine precision.

- Layout-explicit Neumann API: `laplacian_symbol_neumann_node` /
  `_cell`, `solve_poisson_neumann_node` / `_cell`,
  `solve_helmholtz_neumann_node` / `_cell`, `etd1_neumann_node` /
  `_cell`, plus a `centering` argument on the original names.

- `tests/test_dct_i_neumann.py` (44 tests): parity with SciPy, exact
  stencil diagonalization, constant-mode preservation, inverse
  normalization, separable 2D composition, JIT and `grad`
  compatibility, second-order manufactured convergence, and confirmation
  that `centering='cell'` reproduces the pre-1.1.0 symbol exactly.

- **Schnakenberg and Brusselator systems** (`create_schnakenberg_model`,
  `create_brusselator_model`, `schnakenberg_reaction_op`,
  `brusselator_reaction_op`, and the periodic-FFT variants). These
  produce Tables 9 and 10 and were absent from the public repository, so
  two of the paper's three benchmark systems could not previously be
  reproduced from it.

- **15 benchmark and figure scripts** that generate published tables and
  figures but were missing here: the Schnakenberg and Brusselator
  benchmarks, the ablation and FFT-vs-sparse studies, the split
  Gray-Scott legs, the work-precision sweeps, and the pattern gallery,
  attractor divergence and reactor steep-gradient figure generators.
  Their result files, including the `wp_schnakenberg.json` cited in the
  paper, are included.

- `benchmarks/run_all.sh` and `benchmarks/plot_main_figures.py`, the two
  entry points named in the paper's reproduction quickstart. Neither
  existed; `run_all.sh` in particular was named `run_all_benchmarks.sh`
  and covered 10 of the benchmarks, omitting Schnakenberg, Brusselator,
  the work-precision sweeps and the OFAT, ablation, CuPy-FFT and
  JIT-factorial studies.

- `environment-current.yml`, tracking the stack moljax is actively
  developed against, alongside the paper-exact `environment.yml`.

- `rfft2` half-spectrum path for real 2D fields (`use_rfft=True` by
  default on `DiffusionOperator` and the 2D FFT cache), with matching
  ETD and Helmholtz kernels.

### Changed

- `benchmarks/run_all_benchmarks.sh` now forwards to `run_all.sh`.

- `pytest` deselects tests marked `slow` by default; run them with
  `pytest -m slow`, or everything with `pytest -m ""`. Two NILT-bridge
  tests are newly marked `slow` because `compare_nilt_vs_timestepping`
  integrates tens of thousands of ETDRK4 steps through an eager Python
  loop and runs for many minutes to hours. A third test in
  `test_option_a_vs_b.py` carried a `slow` marker that was never
  registered or honoured before, and is now deselected too.

### Notes

- `use_rfft=True` changes floating-point output at the round-off level
  relative to v1.0.0 for 2D periodic problems. Pass `use_rfft=False` for
  bit-comparable behaviour.

- The default Neumann layout change alters results for code that relied
  on `BCType.NEUMANN` meaning cell-centred. The domain length implied by
  `N` and `dx` differs between layouts: `(N-1)·dx` for node-centred
  versus `N·dx` for cell-centred.

## [1.0.0] - 2026-03

Release accompanying the CPC paper. Archival commit `25cd9a3`.
