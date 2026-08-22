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
CROSSOVER_SPEEDUP_RANGE = (0.7, 1.5)


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


def _per_size_reliability(timing_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify CPU/GPU winner determination using their median-plus-IQR bands."""
    cases: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for timing in timing_rows:
        key = (str(timing["workload"]), int(timing["unknowns"]))
        cases.setdefault(key, []).append(timing)

    reliability: list[dict[str, Any]] = []
    for (workload, size), timings in sorted(cases.items()):
        by_backend = {str(timing["backend"]): timing for timing in timings}
        if "cpu" not in by_backend or "gpu" not in by_backend:
            continue

        cpu = by_backend["cpu"]
        gpu = by_backend["gpu"]
        cpu_median = float(cpu["median_seconds"])
        gpu_median = float(gpu["median_seconds"])
        cpu_iqr = float(cpu["iqr_seconds"])
        gpu_iqr = float(gpu["iqr_seconds"])
        fastest, slower = (cpu, gpu) if cpu_median <= gpu_median else (gpu, cpu)
        fastest_median = float(fastest["median_seconds"])
        fastest_iqr = float(fastest["iqr_seconds"])
        slower_median = float(slower["median_seconds"])
        slower_iqr = float(slower["iqr_seconds"])
        robust = fastest_median + fastest_iqr < slower_median - slower_iqr
        speedup = cpu_median / gpu_median
        reliability.append(
            {
                "workload": workload,
                "dimension": int(cpu["dimension"]),
                "grid_size": int(cpu["grid_size"]),
                "unknowns": size,
                "cpu_median_seconds": cpu_median,
                "cpu_iqr_seconds": cpu_iqr,
                "cpu_iqr_band_seconds": [cpu_median - cpu_iqr, cpu_median + cpu_iqr],
                "gpu_median_seconds": gpu_median,
                "gpu_iqr_seconds": gpu_iqr,
                "gpu_iqr_band_seconds": [gpu_median - gpu_iqr, gpu_median + gpu_iqr],
                "gpu_speedup": speedup,
                "fastest_backend": str(fastest["backend"]),
                "classification": "ROBUST" if robust else "NOISE_DOMINATED",
                "crossover_adjacent": (
                    CROSSOVER_SPEEDUP_RANGE[0] <= speedup <= CROSSOVER_SPEEDUP_RANGE[1]
                ),
            }
        )
    return reliability


def _regret_summary(holdouts: Sequence[Any]) -> dict[str, float | int | None]:
    """Return a transparent regret aggregate for a selected holdout subset."""
    regrets = [float(holdout.regret) for holdout in holdouts]
    if not regrets:
        return {"n_holdouts": 0, "mean_regret": None, "max_regret": None}
    return {
        "n_holdouts": len(regrets),
        "mean_regret": float(sum(regrets) / len(regrets)),
        "max_regret": max(regrets),
    }


def _interpretation(
    validation, timing_rows: Sequence[dict[str, Any]], reliability: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize LOO regret together with per-size winner reliability."""
    reliability_by_case = {
        (str(record["workload"]), int(record["unknowns"])): record for record in reliability
    }
    fastest_backend_counts: dict[str, dict[str, int]] = {}
    for record in reliability:
        per_workload = fastest_backend_counts.setdefault(str(record["workload"]), {})
        backend = str(record["fastest_backend"])
        per_workload[backend] = per_workload.get(backend, 0) + 1

    holdout_reliability: list[dict[str, Any]] = []
    robust_crossover_holdouts = []
    for holdout in validation.holdouts:
        record = reliability_by_case.get((holdout.workload, holdout.size))
        if record is None:
            continue
        is_robust_crossover = record["classification"] == "ROBUST" and record["crossover_adjacent"]
        if is_robust_crossover:
            robust_crossover_holdouts.append(holdout)
        holdout_reliability.append(
            {
                "workload": holdout.workload,
                "grid_size": record["grid_size"],
                "unknowns": holdout.size,
                "selected_backend": holdout.selected_backend,
                "actual_fastest_backend": holdout.fastest_backend,
                "heldout_regret": holdout.regret,
                "classification": record["classification"],
                "crossover_adjacent": record["crossover_adjacent"],
            }
        )

    n_comparable = len(reliability)
    crossover_records = [record for record in reliability if record["crossover_adjacent"]]
    n_crossover_adjacent = len(crossover_records)
    n_blowout = n_comparable - n_crossover_adjacent
    n_robust_crossover = sum(record["classification"] == "ROBUST" for record in crossover_records)
    n_noise_dominated_crossover = n_crossover_adjacent - n_robust_crossover

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

    all_regret = _regret_summary(validation.holdouts)
    robust_crossover_regret = _regret_summary(robust_crossover_holdouts)
    mean_regret = validation.overall_mean_regret
    max_regret = validation.overall_max_regret
    easy_decision_regime = (
        mean_regret <= 1.0 + 1.0e-12
        and max_regret <= 1.0 + 1.0e-12
        and n_blowout > n_crossover_adjacent
    )
    if n_robust_crossover == 0:
        caveat = (
            "no robust crossover-adjacent held-out size is available; this workload cannot cleanly "
            "validate dispatch discrimination on the current hardware."
        )
    elif (robust_crossover_regret["max_regret"] or 1.0) > 1.0 + 1.0e-12:
        caveat = (
            "at least one robust crossover-adjacent held-out size is a genuine dispatch miss; "
            "the current calibration is not validated as a reliable crossover predictor."
        )
    elif easy_decision_regime:
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
            "all_holdouts": all_regret,
            "robust_crossover_adjacent_holdouts": robust_crossover_regret,
            "summary": (
                f"all held-outs: mean regret={mean_regret:.6f}; max regret={max_regret:.6f}; "
                f"robust crossover-adjacent holdouts={n_robust_crossover}"
            ),
        },
        "decision_regime": {
            "crossover_adjacent_factor": CROSSOVER_ADJACENT_FACTOR,
            "crossover_speedup_range": list(CROSSOVER_SPEEDUP_RANGE),
            "heldout_sizes": len(validation.holdouts),
            "backend_comparable_holdout_sizes": n_comparable,
            "crossover_adjacent_holdout_sizes": n_crossover_adjacent,
            "robust_crossover_adjacent_holdout_sizes": n_robust_crossover,
            "noise_dominated_crossover_adjacent_holdout_sizes": n_noise_dominated_crossover,
            "blowout_holdout_sizes": n_blowout,
            "fastest_backend_counts": fastest_backend_counts,
            "summary": (
                f"{n_crossover_adjacent} of {n_comparable} backend-comparable held-out sizes are "
                f"crossover-adjacent (CPU/GPU speedup in {CROSSOVER_SPEEDUP_RANGE[0]:.1f}x--"
                f"{CROSSOVER_SPEEDUP_RANGE[1]:.1f}x): {n_robust_crossover} ROBUST and "
                f"{n_noise_dominated_crossover} NOISE_DOMINATED; {n_blowout} are blowout decisions."
            ),
        },
        "holdout_reliability": holdout_reliability,
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
    reliability = _per_size_reliability(timing_rows)
    payload = {
        "schema": "device_dispatch_cv_v1",
        "source": "locally measured moljax two-field FFT Helmholtz solves",
        "timing_protocol": "two warmups excluded; synchronized median and IQR",
        "devices": {backend: str(device) for backend, device in devices.items()},
        "device_primitives": {
            backend: asdict(measurement) for backend, measurement in primitive_measurements.items()
        },
        "config": {
            "sizes_1d": list(sizes_1d),
            "sizes_2d": list(sizes_2d),
            "repetitions": repetitions,
            "warmups": 2,
            "crossover_speedup_range": list(CROSSOVER_SPEEDUP_RANGE),
        },
        "timings": timing_rows,
        "per_size_reliability": reliability,
        "rows": _validation_rows(validation),
        "overall": {
            "mean_regret": validation.overall_mean_regret,
            "max_regret": validation.overall_max_regret,
        },
        "interpretation": _interpretation(validation, timing_rows, reliability),
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
