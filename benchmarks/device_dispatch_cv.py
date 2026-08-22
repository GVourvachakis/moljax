#!/usr/bin/env python3
# ruff: noqa: E402
"""Fresh leave-one-size-out validation for moljax FFT Helmholtz solves."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moljax.core.fft_solvers import (
    laplacian_symbol_1d,
    laplacian_symbol_2d,
    solve_helmholtz_1d,
    solve_helmholtz_2d,
)
from moljax.core.gpu_benchmarks import DeviceInfo, measure_device_primitives
from moljax.experimental.device_dispatch import (
    DevicePrimitives,
    DispatchPolicy,
    FFTPrimitiveCostModel,
    TimingSample,
    leave_one_size_out,
)

try:
    from benchmarks.benchmark_utils import time_call
except ModuleNotFoundError:
    from benchmark_utils import time_call


RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_OUTPUT = RESULTS_DIR / "device_dispatch_cv.json"
DEFAULT_SIZES_1D = (256, 512, 1024, 2048)
DEFAULT_SIZES_2D = (64, 128, 256, 512)


def available_devices() -> dict[str, Any]:
    """Return moljax-detected CPU/GPU devices that can be benchmarked."""
    info = DeviceInfo.detect()
    devices: dict[str, Any] = {"cpu": jax.devices("cpu")[0]}
    if info.has_gpu():
        devices["gpu"] = jax.devices("gpu")[0]
    return devices


def _two_field_helmholtz_call(device: Any, dimension: int, size: int):
    """Build one device-pinned two-field periodic FFT preconditioner call."""
    with jax.default_device(device):
        if dimension == 1:
            dx = 1.0 / size
            symbol = jax.device_put(laplacian_symbol_1d(size, dx), device)
            x = jnp.arange(size, dtype=jnp.float64) / size
            rhs_u = jnp.sin(2.0 * jnp.pi * x)
            rhs_v = jnp.cos(4.0 * jnp.pi * x)

            def solve(u: jax.Array, v: jax.Array, laplacian: jax.Array):
                return (
                    solve_helmholtz_1d(u, laplacian, 0.02, 0.01),
                    solve_helmholtz_1d(v, laplacian, 0.02, 0.02),
                )

        elif dimension == 2:
            dx = 1.0 / size
            symbol = jax.device_put(laplacian_symbol_2d(size, size, dx, dx), device)
            coordinates = jnp.arange(size, dtype=jnp.float64) / size
            x, y = jnp.meshgrid(coordinates, coordinates, indexing="xy")
            rhs_u = jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)
            rhs_v = jnp.cos(4.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)

            def solve(u: jax.Array, v: jax.Array, laplacian: jax.Array):
                return (
                    solve_helmholtz_2d(u, laplacian, 0.02, 0.01),
                    solve_helmholtz_2d(v, laplacian, 0.02, 0.02),
                )

        else:
            raise ValueError("dimension must be 1 or 2")

        compiled = jax.jit(solve)
        return lambda: compiled(rhs_u, rhs_v, symbol)


def measure_helmholtz_timing(
    *,
    device: Any,
    backend: str,
    dimension: int,
    size: int,
    repetitions: int,
    extra_warmups: int = 0,
) -> tuple[TimingSample, dict[str, Any]]:
    """Time the same two-field FFT diffusion preconditioner on one device."""
    call = _two_field_helmholtz_call(device, dimension, size)
    if extra_warmups < 0:
        raise ValueError("extra_warmups must be non-negative")
    for _ in range(extra_warmups):
        jax.tree_util.tree_map(lambda value: value.block_until_ready(), call())
    _, median_seconds, iqr_seconds, _ = time_call(call, n_reps=repetitions)
    workload = f"fft_helmholtz_{dimension}d"
    unknowns = 2 * size if dimension == 1 else 2 * size * size
    sample = TimingSample(workload, backend, unknowns, median_seconds)
    return sample, {
        "workload": workload,
        "dimension": dimension,
        "grid_size": size,
        "unknowns": unknowns,
        "backend": backend,
        "device": str(device),
        "median_seconds": median_seconds,
        "iqr_seconds": iqr_seconds,
        "repetitions": repetitions,
        "warmups": 1 + extra_warmups,
    }


def _json_float(value: float) -> float | None:
    """Convert non-finite floating diagnostics to JSON null."""
    return value if jnp.isfinite(value) else None


def _validation_rows(validation) -> list[dict[str, Any]]:
    """Return transparent holdout and aggregate rows for persisted output."""
    rows: list[dict[str, Any]] = []
    for result in validation.holdouts:
        rows.append(
            {
                "row_type": "holdout",
                "workload": result.workload,
                "M": result.size,
                "predicted_backend": result.selected_backend,
                "actual_fastest_backend": result.fastest_backend,
                "heldout_regret": result.regret,
                "predicted_seconds": _json_float(result.predicted_seconds),
                "actual_seconds": result.actual_seconds,
            }
        )
    for result in validation.model_summaries:
        rows.append(
            {
                "row_type": "model_summary",
                "workload": result.workload,
                "backend": result.backend,
                "n_sizes": result.n_sizes,
                "q2": _json_float(result.q2),
                "l2_error_seconds": result.l2_error_seconds,
            }
        )
    for result in validation.case_summaries:
        rows.append(
            {
                "row_type": "case_summary",
                "workload": result.workload,
                "n_sizes": result.n_sizes,
                "mean_regret": result.mean_regret,
                "max_regret": result.max_regret,
            }
        )
    rows.append(
        {
            "row_type": "overall_summary",
            "n_sizes": len(validation.holdouts),
            "mean_regret": validation.overall_mean_regret,
            "max_regret": validation.overall_max_regret,
        }
    )
    return rows


def run_device_dispatch_cv(
    *,
    output: Path = DEFAULT_OUTPUT,
    sizes_1d: Sequence[int] = DEFAULT_SIZES_1D,
    sizes_2d: Sequence[int] = DEFAULT_SIZES_2D,
    repetitions: int = 21,
) -> dict[str, Any]:
    """Generate a new leave-one-size-out report from locally timed solves."""
    devices = available_devices()
    primitive_measurements = {
        backend: measure_device_primitives(device) for backend, device in devices.items()
    }
    primitives = {
        backend: DevicePrimitives(measurement.fft_seconds, measurement.launch_seconds)
        for backend, measurement in primitive_measurements.items()
    }
    samples: list[TimingSample] = []
    timing_rows: list[dict[str, Any]] = []
    for dimension, sizes in ((1, sizes_1d), (2, sizes_2d)):
        for size in sizes:
            for backend, device in devices.items():
                sample, row = measure_helmholtz_timing(
                    device=device,
                    backend=backend,
                    dimension=dimension,
                    size=size,
                    repetitions=repetitions,
                    extra_warmups=1,
                )
                samples.append(sample)
                timing_rows.append(row)
    validation = leave_one_size_out(
        samples,
        models={backend: FFTPrimitiveCostModel() for backend in devices},
        primitives=primitives,
        policy=DispatchPolicy(fallback_backend="cpu"),
    )
    payload = {
        "schema": "device_dispatch_cv_v1",
        "source": "locally measured moljax two-field FFT Helmholtz solves",
        "timing_protocol": "two warmups excluded; synchronized median and IQR",
        "devices": {backend: str(device) for backend, device in devices.items()},
        "device_primitives": {
            backend: asdict(measurement) for backend, measurement in primitive_measurements.items()
        },
        "timings": timing_rows,
        "rows": _validation_rows(validation),
        "overall": {
            "mean_regret": validation.overall_mean_regret,
            "max_regret": validation.overall_max_regret,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Run the local calibration and print its held-out regret summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=21)
    arguments = parser.parse_args()
    result = run_device_dispatch_cv(output=arguments.output, repetitions=arguments.repetitions)
    print(
        "LOO dispatch: "
        f"mean regret={result['overall']['mean_regret']:.6f}, "
        f"max regret={result['overall']['max_regret']:.6f}"
    )


if __name__ == "__main__":
    main()
