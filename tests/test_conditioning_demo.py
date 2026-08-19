"""CPU smoke tests for the FFT-preconditioned conditioning decision demo."""

from __future__ import annotations

import jax

from benchmarks.conditioning_decision_demo import DemoConfig, run_decision_demo


def test_small_diffusion_decision_demo():
    """A visited diffusion-dominated state passes the adjoint and verdict gates."""
    with jax.enable_x64(True):
        result = run_decision_demo(
            DemoConfig(
                nx=8,
                ny=8,
                n_states=1,
                n_angles=4,
                fov_max_iters=12,
                arnoldi_steps=3,
                pseudospectrum_points=3,
                overhead_runs=5,
                max_newton_iters=10,
                max_krylov_iters=18,
            )
        )

    record = result["states"][0]
    assert record["status"] == "completed"
    assert record["implicit_step"]["converged"] is True
    assert record["adjoint_identity"] <= 1.0e-8
    assert record["assessment"]["verdict"] == "adequate"
