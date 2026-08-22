"""Guarded smoke coverage for the local GPU dispatch benchmark."""

from pathlib import Path

import jax
import pytest

from benchmarks.device_dispatch_cv import _per_size_reliability


def test_per_size_reliability_marks_overlapping_iqr_bands_as_noise_dominated():
    """A crossover winner is useful only when the timing bands are separable."""
    rows = [
        {
            "workload": "fft_helmholtz_2d",
            "dimension": 2,
            "grid_size": 96,
            "unknowns": 18_432,
            "backend": "cpu",
            "median_seconds": 1.0,
            "iqr_seconds": 0.05,
        },
        {
            "workload": "fft_helmholtz_2d",
            "dimension": 2,
            "grid_size": 96,
            "unknowns": 18_432,
            "backend": "gpu",
            "median_seconds": 1.2,
            "iqr_seconds": 0.05,
        },
        {
            "workload": "fft_helmholtz_2d",
            "dimension": 2,
            "grid_size": 104,
            "unknowns": 21_632,
            "backend": "cpu",
            "median_seconds": 1.0,
            "iqr_seconds": 0.2,
        },
        {
            "workload": "fft_helmholtz_2d",
            "dimension": 2,
            "grid_size": 104,
            "unknowns": 21_632,
            "backend": "gpu",
            "median_seconds": 1.1,
            "iqr_seconds": 0.2,
        },
    ]

    reliability = _per_size_reliability(rows)

    assert [record["classification"] for record in reliability] == ["ROBUST", "NOISE_DOMINATED"]
    assert all(record["crossover_adjacent"] for record in reliability)


@pytest.mark.slow
def test_gpu_dispatch_benchmark_smoke(tmp_path: Path):
    """A tiny local benchmark should emit fresh held-out records on a GPU."""
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("GPU unavailable")
    from benchmarks.device_dispatch_cv import run_device_dispatch_cv

    output = tmp_path / "dispatch.json"
    result = run_device_dispatch_cv(
        output=output,
        sizes_1d=(32, 64, 128),
        sizes_2d=(8, 16, 32),
        repetitions=3,
    )
    assert output.is_file()
    assert result["overall"]["mean_regret"] >= 1.0
    assert any(row["row_type"] == "holdout" for row in result["rows"])
    interpretation = result["interpretation"]
    decision_regime = interpretation["decision_regime"]
    assert decision_regime["backend_comparable_holdout_sizes"] == (
        decision_regime["crossover_adjacent_holdout_sizes"]
        + decision_regime["blowout_holdout_sizes"]
    )
    assert len(interpretation["model_fit"]) == 4
