#!/usr/bin/env python3
# ruff: noqa: E402
"""CPU/GPU scaling study for the moljax two-field FFT diffusion preconditioner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

try:
    from benchmarks.device_dispatch_cv import available_devices, measure_helmholtz_timing
except ModuleNotFoundError:
    from device_dispatch_cv import available_devices, measure_helmholtz_timing

RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_OUTPUT = RESULTS_DIR / "scaling_reproduction.json"
DEFAULT_SIZES = (64, 128, 256, 512, 1024)
SAFE_MEMORY_MIB = 20 * 1024


def gpu_memory_mib() -> int | None:
    """Return total reported GPU memory use, or ``None`` when unavailable."""
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    try:
        return int(values[0]) if values else None
    except ValueError:
        return None


def memory_probe(
    *,
    device: Any,
    size: int = 256,
    repetitions: int = 3,
    target_size: int = 1024,
) -> dict[str, Any]:
    """Measure a resident-memory proxy and conservatively project a target size."""
    before_mib = gpu_memory_mib()
    _, timing = measure_helmholtz_timing(
        device=device,
        backend="gpu",
        dimension=2,
        size=size,
        repetitions=repetitions,
    )
    after_mib = gpu_memory_mib()
    measured_increment = max(0, (after_mib or 0) - (before_mib or 0))
    conservative_working_mib = 32.0 * size * size * 8.0 / (1024.0 * 1024.0)
    projected_increment = max(measured_increment, conservative_working_mib)
    projected_target_mib = (before_mib or 0) + projected_increment * (target_size / size) ** 2
    return {
        "probe_grid_size": size,
        "probe_timing": timing,
        "memory_before_mib": before_mib,
        "memory_after_mib": after_mib,
        "resident_increment_mib": measured_increment,
        "conservative_working_estimate_mib": conservative_working_mib,
        "projected_target_mib": projected_target_mib,
        "target_grid_size": target_size,
        "safe_budget_mib": SAFE_MEMORY_MIB,
        "measurement_note": "post-warmup resident-memory proxy; projection includes a conservative workspace factor",
    }


def _safe_sizes(sizes: Sequence[int], probe: dict[str, Any]) -> tuple[int, ...]:
    """Keep only sizes whose quadratic memory projection stays in the budget."""
    base_mib = probe["memory_before_mib"] or 0
    probe_size = probe["probe_grid_size"]
    increment = max(
        probe["resident_increment_mib"],
        probe["conservative_working_estimate_mib"],
    )
    return tuple(
        size for size in sizes if base_mib + increment * (size / probe_size) ** 2 <= SAFE_MEMORY_MIB
    )


def _work_normalized_seconds(record: dict[str, Any], backend: str) -> float:
    """Return time normalized by the two-dimensional FFT work proxy."""
    import math

    size = record["grid_size"]
    work = size * size * math.log(size * size)
    return record[backend]["median_seconds"] / work


def run_scaling_reproduction(
    *,
    output: Path = DEFAULT_OUTPUT,
    sizes: Sequence[int] = DEFAULT_SIZES,
    repetitions: int = 7,
) -> dict[str, Any]:
    """Measure CPU/GPU size scaling after a GPU memory safety probe."""
    devices = available_devices()
    if "gpu" not in devices:
        payload = {
            "schema": "scaling_reproduction_v1",
            "status": "gpu_unavailable",
            "devices": {backend: str(device) for backend, device in devices.items()},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    probe = memory_probe(device=devices["gpu"], target_size=max(sizes))
    safe_sizes = _safe_sizes(sizes, probe)
    records: list[dict[str, Any]] = []
    for size in safe_sizes:
        per_backend: dict[str, dict[str, Any]] = {}
        for backend, device in devices.items():
            _, timing = measure_helmholtz_timing(
                device=device,
                backend=backend,
                dimension=2,
                size=size,
                repetitions=repetitions,
            )
            per_backend[backend] = timing
        cpu_seconds = per_backend["cpu"]["median_seconds"]
        gpu_seconds = per_backend["gpu"]["median_seconds"]
        records.append(
            {
                "grid_size": size,
                "unknowns": 2 * size * size,
                "cpu": per_backend["cpu"],
                "gpu": per_backend["gpu"],
                "gpu_speedup": cpu_seconds / gpu_seconds,
            }
        )
    speedups = [record["gpu_speedup"] for record in records]
    cpu_normalized = [_work_normalized_seconds(record, "cpu") for record in records]
    gpu_normalized = [_work_normalized_seconds(record, "gpu") for record in records]
    crossover = next(
        (record["grid_size"] for record in records if record["gpu_speedup"] > 1.0), None
    )
    payload = {
        "schema": "scaling_reproduction_v1",
        "source": "locally measured moljax two-field FFT Helmholtz solves",
        "timing_protocol": "one warmup excluded; synchronized median and IQR",
        "hardware": {
            "cpu": str(devices["cpu"]),
            "gpu": str(devices["gpu"]),
            "comparison_gpu": "RTX 5060",
            "comparison_note": "absolute speedups are hardware-specific; only size-scaling trends are compared",
        },
        "memory_probe": probe,
        "requested_grid_sizes": list(sizes),
        "safe_grid_sizes": list(safe_sizes),
        "records": records,
        "trend": {
            "gpu_crossover_grid_size": crossover,
            "speedup_grows_from_first_to_last": len(speedups) > 1 and speedups[-1] > speedups[0],
            "cpu_seconds_over_n2_log_n": cpu_normalized,
            "gpu_seconds_over_n2_log_n": gpu_normalized,
            "cpu_n2_log_n_normalized_spread": max(cpu_normalized) / min(cpu_normalized),
            "gpu_n2_log_n_normalized_spread": max(gpu_normalized) / min(gpu_normalized),
            "largest_safe_grid_size": safe_sizes[-1] if safe_sizes else None,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Run the guarded local CPU/GPU scaling study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=7)
    arguments = parser.parse_args()
    result = run_scaling_reproduction(output=arguments.output, repetitions=arguments.repetitions)
    summary = result["trend"] if "trend" in result else {"status": result["status"]}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
