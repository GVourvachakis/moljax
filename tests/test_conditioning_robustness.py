"""Regression tests for conditioning-diagnostic failure modes.

Each test here pins a defect that would otherwise return a confident but
wrong answer:

  1. an enclosing disk that does not enclose its own boundary at small scale,
  2. an unconverged eigensolve accepted as a numerical-range support point,
  3. a diagnostic run that reports success after the implicit solve failed.

All three corrupt the adequacy verdict rather than raising, so they are the
failure modes worth gating in CI.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmarks.conditioning_decision_demo import (  # noqa: E402
    DemoConfig,
    run_decision_demo,
)
from moljax.conditioning._geometry import _smallest_enclosing_disk  # noqa: E402
from moljax.conditioning.field_of_values import numerical_range  # noqa: E402


class TestEnclosingDiskScaleInvariance:
    """The disk must enclose its points at every magnitude, not just near one."""

    @pytest.mark.parametrize("exponent", [-12, -10, -8, -4, 0, 4, 8, 12])
    def test_disk_encloses_boundary(self, exponent: int) -> None:
        scale = 10.0**exponent
        points = np.exp(2j * np.pi * np.arange(7) / 7) * scale
        center, radius = _smallest_enclosing_disk(points)
        worst = max(abs(point - center) for point in points)
        # A disk that misses its own boundary understates the radius, and with
        # it the disk rate the verdict is read from.
        assert worst <= radius * (1.0 + 1.0e-9)

    @pytest.mark.parametrize("exponent", [-12, -6, 0, 6, 12])
    def test_radius_scales_linearly(self, exponent: int) -> None:
        scale = 10.0**exponent
        points = np.exp(2j * np.pi * np.arange(9) / 9) * scale
        _, radius = _smallest_enclosing_disk(points)
        assert radius == pytest.approx(scale, rel=1.0e-9)

    def test_collinear_points_still_enclosed(self) -> None:
        points = np.asarray([0.0 + 0.0j, 1e-10 + 0.0j, 2e-10 + 0.0j])
        center, radius = _smallest_enclosing_disk(points)
        assert max(abs(p - center) for p in points) <= radius * (1.0 + 1.0e-9)


class TestSupportConvergenceIsChecked:
    """An unconverged support point must be refused, not returned."""

    @staticmethod
    def _diagonal(n: int):
        diag = jnp.asarray(np.linspace(0.01, 1.2, n))
        return (lambda v: diag * v), (lambda v: jnp.conj(diag) * v)

    def test_truncated_budget_raises(self) -> None:
        matvec, adjoint = self._diagonal(120)
        with pytest.raises(RuntimeError, match="did not converge"):
            numerical_range(matvec, adjoint, 120, n_angles=8, max_iters=1)

    @pytest.mark.slow
    def test_converged_budget_recovers_known_interval(self) -> None:
        n = 120
        matvec, adjoint = self._diagonal(n)
        result = numerical_range(matvec, adjoint, n, n_angles=24, max_iters=120)
        # W(A) of a real diagonal operator is the interval [0.01, 1.2].
        assert result.radius == pytest.approx(0.595, abs=5.0e-3)
        assert result.center.real == pytest.approx(0.605, abs=5.0e-3)
        assert result.max_support_residual < 1.0e-3
        assert not result.origin_enclosed


class TestFailedImplicitStepIsNotReportedComplete:
    """A failed Newton solve must not be diagnosed or reported as success."""

    @pytest.mark.slow
    def test_nonconverged_newton_marks_run_failed(self, tmp_path) -> None:
        result = run_decision_demo(
            DemoConfig(
                nx=8,
                ny=8,
                dt=2.0,
                n_states=1,
                n_angles=4,
                fov_max_iters=60,
                arnoldi_steps=3,
                pseudospectrum_points=3,
                overhead_runs=2,
                max_newton_iters=0,
                max_krylov_iters=18,
                figure_dir=str(tmp_path),
            )
        )
        assert result["status"] == "failed"
        assert result["implicit_step_failures"]
        assert result["implicit_step_failures"][0]["converged"] is False
        # No verdict may be emitted for a state the solver never reached.
        assert result["states"] == []
