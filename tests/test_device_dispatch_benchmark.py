"""Guarded smoke coverage for the local GPU dispatch benchmark."""

from pathlib import Path

import jax
import pytest


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
