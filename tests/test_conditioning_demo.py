"""CPU smoke tests for the FFT-preconditioned conditioning decision demo."""

from __future__ import annotations

import jax

from benchmarks.conditioning_decision_demo import DemoConfig, run_decision_demo


def test_small_diffusion_decision_demo(tmp_path):
    """Visited states pass the adjoint gate and cover both decision verdicts."""
    with jax.enable_x64(True):
        result = run_decision_demo(
            DemoConfig(
                nx=8,
                ny=8,
                dt=2.0,
                n_states=1,
                n_angles=4,
                fov_max_iters=12,
                arnoldi_steps=3,
                pseudospectrum_points=3,
                overhead_runs=5,
                max_newton_iters=10,
                max_krylov_iters=18,
                figure_dir=str(tmp_path),
            )
        )

    assert all(record["status"] == "completed" for record in result["states"])
    assert all(record["implicit_step"]["converged"] is True for record in result["states"])
    assert all(record["adjoint_identity"] <= 1.0e-8 for record in result["states"])
    assert {record["assessment"]["verdict"] for record in result["states"]} == {
        "adequate",
        "investigate",
    }
    assert len(result["figures"]) == 8
