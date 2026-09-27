"""Smoke tests for the experimental two-regime Brusselator conditioning study."""

from __future__ import annotations

import math
from dataclasses import replace

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import pytest

from benchmarks import brusselator_conditioning as benchmark
from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    assess_brusselator_state,
    build_brusselator_system,
    sampled_visited_states,
    visited_states,
)
from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
    dense_padded_preconditioned_operator,
)


def _homogeneous_state(regime, grid):
    model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
    state = model.apply_bcs(
        {
            "u": jax.numpy.full(
                (grid.ny_total, grid.nx_total), regime.a, dtype=jax.numpy.float64
            ),
            "v": jax.numpy.full(
                (grid.ny_total, grid.nx_total), regime.b / regime.a, dtype=jax.numpy.float64
            ),
        },
        0.0,
    )
    return state, model, fft_cache, diffusivities


@pytest.fixture(scope="module")
def tiny_fft_records():
    """Evaluate the minimal FFT-only two-regime smoke configuration once."""
    records = {}
    for seed, regime in enumerate((HOPF_REGIME, TURING_REGIME), start=1):
        grid = Grid2D.uniform(8, 8, 0.0, regime.domain_length, 0.0, regime.domain_length)
        state = visited_states(
            regime,
            grid=grid,
            n_steps=1,
            dt=0.1,
            perturbation=1.0e-3,
            seed=seed,
        )[0]
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        records[regime.name] = assess_brusselator_state(
            state,
            model,
            fft_cache,
            diffusivities,
            0.1,
            regime,
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=3,
            seed=seed,
        )
    return records


@pytest.mark.slow
def test_hopf_visited_state_passes_adjoint_gate_and_has_a_verdict(tiny_fft_records):
    """A tiny FFT-preconditioned Hopf state is valid input to the toolbox."""
    record = tiny_fft_records["hopf"]
    assert record["status"] == "completed"
    assert record["adjoint_error"] <= 1.0e-8
    assert record["verdict"] in {"adequate", "investigate", "indeterminate"}


@pytest.mark.slow
def test_both_regimes_record_structural_discriminators(tiny_fft_records):
    """The outcome is data, but both physical-regime record fields must exist."""
    for regime in ("hopf", "turing"):
        record = tiny_fft_records[regime]
        assert record["status"] == "completed"
        assert record["adjoint_error"] <= 1.0e-8
        assert isinstance(record["origin_enclosed"], bool)
        assert math.isfinite(record["fov_imaginary_extent"])
        assert record["fov_imaginary_extent"] >= 0.0


@pytest.mark.slow
def test_developed_hopf_sample_leaves_the_fixed_point_and_passes_adjoint_gate():
    """A late sampled Hopf state is developed rather than a seed perturbation."""
    perturbation = 1.0e-3
    grid = Grid2D.uniform(
        8,
        8,
        0.0,
        HOPF_REGIME.domain_length,
        0.0,
        HOPF_REGIME.domain_length,
    )
    samples = sampled_visited_states(
        HOPF_REGIME,
        grid=grid,
        sample_steps=(1, 5, 10),
        dt=1.0,
        perturbation=perturbation,
        seed=20260822,
        nk_params=NKParams(
            max_newton_iters=15,
            max_krylov_iters=100,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        ),
    )
    late = samples[-1]
    model, fft_cache, diffusivities = build_brusselator_system(HOPF_REGIME, grid)
    assessment = assess_brusselator_state(
        late.state,
        model,
        fft_cache,
        diffusivities,
        1.0,
        HOPF_REGIME,
        n_angles=3,
        fov_max_iters=4,
        arnoldi_steps=3,
        seed=20260822,
    )

    departure = max(late.developedness.values())
    assert departure > 20.0 * perturbation
    assert assessment["status"] == "completed"
    assert assessment["adjoint_error"] <= 1.0e-8


@pytest.mark.slow
def test_fourier_weyl_bound_certifies_a_small_homogeneous_turing_state():
    """The integrated full-operator bound unlocks adequacy only with all gates clear."""
    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.01,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        compute_lobpcg_upper_estimate=True,
        seed=20260821,
    )
    certificate = assessment["fourier_weyl_ghost_lower_bound"]
    interior = np.asarray(state["u"])[1:-1, 1:-1]
    interior_v = np.asarray(state["v"])[1:-1, 1:-1]
    dense = float(
        np.linalg.svd(
            dense_padded_preconditioned_operator(
                interior,
                interior_v,
                du=TURING_REGIME.du,
                dv=TURING_REGIME.dv,
                beta=TURING_REGIME.b,
                dt=0.01,
            ),
            compute_uv=False,
        )[-1]
    )
    assert assessment["verdict"] == "adequate"
    assert assessment["epsilon_zero_full_operator_evidence"] is True
    assert certificate["status"] == "clears_adequacy_gate"
    assert certificate["full_lower_bound"] >= 0.1
    assert certificate["full_lower_bound"] <= dense + 5.0e-13
    assert assessment["lobpcg_sigma_min_upper_estimate"] is not None


@pytest.mark.slow
def test_valid_weak_fourier_weyl_bound_preserves_provisional(monkeypatch):
    """A valid but insufficient lower bound cannot promote a provisional reading."""
    import moljax.experimental.brusselator_conditioning as conditioning

    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    genuine = conditioning._fourier_weyl_bound(state, model, TURING_REGIME, 0.01)
    weak_selected = replace(genuine.selected, full_lower_bound=0.05)
    monkeypatch.setattr(
        conditioning,
        "_fourier_weyl_bound",
        lambda *_args, **_kwargs: replace(genuine, selected=weak_selected),
    )
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.01,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        seed=20260821,
    )
    certificate = assessment["fourier_weyl_ghost_lower_bound"]
    assert genuine.selected.full_lower_bound >= 0.1
    assert assessment["verdict"] == "provisional"
    assert assessment["epsilon_zero_full_operator_evidence"] is False
    assert certificate["status"] == "valid_but_below_adequacy_gate"
    assert certificate["full_lower_bound"] == pytest.approx(0.05)


@pytest.mark.slow
def test_origin_enclosure_remains_indeterminate_despite_a_valid_bound():
    """The full-operator lower bound cannot override an origin-enclosed FOV."""
    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state = visited_states(
        TURING_REGIME,
        grid=grid,
        n_steps=1,
        dt=0.2,
        perturbation=0.8,
        seed=20260928,
    )[0]
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.2,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        seed=20260928,
    )
    assert assessment["fourier_weyl_ghost_lower_bound"]["full_lower_bound"] >= 0.0
    assert assessment["origin_enclosed"] is True
    assert assessment["verdict"] == "indeterminate"


def test_v4_source_cache_rejects_a_foreign_generation_fingerprint(tmp_path):
    """Source reuse fails closed rather than crossing Brusselator configurations."""
    config = benchmark._config(
        "screen_64",
        nx=4,
        ny=4,
        n_states=1,
        source_state_cache_dir=str(tmp_path),
    )
    fingerprint = benchmark._source_state_fingerprint(config, TURING_REGIME, (1,), 7)
    state = {
        "u": jax.numpy.ones((6, 6), dtype=jax.numpy.float64),
        "v": jax.numpy.full((6, 6), 1.8, dtype=jax.numpy.float64),
    }
    benchmark._persist_source_states(config, TURING_REGIME, fingerprint, [state])
    foreign = {**fingerprint, "seed": 8}
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        benchmark._load_cached_states(config, TURING_REGIME, foreign)
