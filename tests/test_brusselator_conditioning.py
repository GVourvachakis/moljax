"""Smoke tests for the experimental two-regime Brusselator conditioning study."""

from __future__ import annotations

import math

import jax
import pytest

from moljax.core.grid import Grid2D
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    assess_brusselator_state,
    build_brusselator_system,
    visited_states,
)


@pytest.fixture(scope="module")
def tiny_fft_records():
    """Evaluate the minimal FFT-only two-regime smoke configuration once."""
    records = {}
    with jax.enable_x64(True):
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
