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
CROSSOVER_ADJACENT_FACTOR = 1.5


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


def _interpretation(validation, timing_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether held-out regret tests a non-trivial backend choice."""
    cases: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for timing in timing_rows:
        key = (str(timing["workload"]), int(timing["grid_size"]))
        cases.setdefault(key, []).append(timing)

    comparable_cases: list[tuple[str, int, str, float, bool]] = []
    fastest_backend_counts: dict[str, dict[str, int]] = {}
    for (workload, size), timings in sorted(cases.items()):
        if len(timings) < 2:
            continue
        ranked = sorted(timings, key=lambda timing: float(timing["median_seconds"]))
        fastest = ranked[0]
        fastest_time = float(fastest["median_seconds"])
        second_time = float(ranked[1]["median_seconds"])
        second_best_factor = second_time / fastest_time
        is_crossover_adjacent = second_best_factor <= CROSSOVER_ADJACENT_FACTOR
        comparable_cases.append(
            (workload, size, str(fastest["backend"]), second_best_factor, is_crossover_adjacent)
        )
        per_workload = fastest_backend_counts.setdefault(workload, {})
        backend = str(fastest["backend"])
        per_workload[backend] = per_workload.get(backend, 0) + 1

    n_comparable = len(comparable_cases)
    n_crossover_adjacent = sum(case[-1] for case in comparable_cases)
    n_blowout = n_comparable - n_crossover_adjacent
    model_fit: list[dict[str, Any]] = []
    n_negative_q2 = 0
    for summary in sorted(
        validation.model_summaries,
        key=lambda item: (item.workload, item.backend),
    ):
        q2 = _json_float(summary.q2)
        negative_q2 = q2 is not None and q2 < 0.0
        n_negative_q2 += int(negative_q2)
        model_fit.append(
            {
                "workload": summary.workload,
                "backend": summary.backend,
                "q2": q2,
                "fit_status": (
                    "cost model does not fit; predictions unreliable for this backend/workload"
                    if negative_q2
                    else "non-negative q2 on this size ladder"
                ),
            }
        )

    mean_regret = validation.overall_mean_regret
    max_regret = validation.overall_max_regret
    easy_decision_regime = (
        mean_regret <= 1.0 + 1.0e-12
        and max_regret <= 1.0 + 1.0e-12
        and n_blowout > n_crossover_adjacent
    )
    if easy_decision_regime:
        caveat = (
            "low regret reflects an easy decision regime, not a validated predictor; "
            "a workload spanning the CPU/GPU crossover is needed to test the dispatcher's discrimination."
        )
    elif n_comparable == 0:
        caveat = "no size has more than one timed backend, so dispatch discrimination is untested."
    else:
        caveat = (
            "held-out regret includes crossover-adjacent decisions, but fit diagnostics should be considered "
            "alongside regret when judging predictive quality."
        )

    return {
        "heldout_regret": {
            "mean": mean_regret,
            "max": max_regret,
            "summary": (
                f"mean held-out regret={mean_regret:.6f}; max held-out regret={max_regret:.6f}"
            ),
        },
        "decision_regime": {
            "crossover_adjacent_factor": CROSSOVER_ADJACENT_FACTOR,
            "heldout_sizes": len(validation.holdouts),
            "backend_comparable_holdout_sizes": n_comparable,
            "crossover_adjacent_holdout_sizes": n_crossover_adjacent,
            "blowout_holdout_sizes": n_blowout,
            "fastest_backend_counts": fastest_backend_counts,
            "summary": (
                f"{n_crossover_adjacent} of {n_comparable} backend-comparable held-out sizes are "
                f"crossover-adjacent (second-best within {CROSSOVER_ADJACENT_FACTOR:.1f}x); "
                f"{n_blowout} are blowout decisions."
            ),
        },
        "model_fit": model_fit,
        "negative_q2_model_count": n_negative_q2,
        "caveat": caveat,
    }


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
        "interpretation": _interpretation(validation, timing_rows),
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
