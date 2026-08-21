#!/usr/bin/env python3
"""Conditioning study on developed periodic Brusselator trajectories.

This follow-up to ``brusselator_conditioning.py`` reaches nonlinear states
well beyond the initial 1e-3 perturbation before applying the same
field-of-values, Arnoldi, and counted-GMRES diagnostics.  It remains a 64²
screen, below the paper's 256² target, and reports that scope explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, NamedTuple

import jax

from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    BrusselatorRegime,
    assess_brusselator_state,
    build_brusselator_system,
    measure_brusselator_gmres,
    sampled_visited_states,
)


class DevelopedBrusselatorStudyConfig(NamedTuple):
    """A tractable developed-state trajectory and diagnostic schedule.

    The 64² grid and backward-Euler ``dt=1`` make a full ``t=200`` Turing
    trajectory tractable while retaining shipped moljax dynamics.  Hopf
    develops much sooner and is sampled through ``t=20``; Turing is sampled
    during growth, established pattern formation, and at its paper horizon.
    """

    nx: int = 64
    ny: int = 64
    dt: float = 1.0
    perturbation: float = 1.0e-3
    hopf_sample_steps: tuple[int, ...] = (1, 10, 20)
    turing_sample_steps: tuple[int, ...] = (80, 120, 200)
    seed: int = 20260822
    n_angles: int = 4
    fov_max_iters: int = 8
    arnoldi_steps: int = 6
    max_newton_iters: int = 15
    max_krylov_iters: int = 100
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    output_path: str = "benchmarks/results/brusselator_conditioning_developed.json"


def _distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count each valid decision-procedure outcome."""
    values = {"adequate": 0, "investigate": 0, "indeterminate": 0, "skipped": 0}
    for record in records:
        values[record["verdict"]] = values.get(record["verdict"], 0) + 1
    return values


def _fft_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return completed FFT-preconditioned rows in trajectory order."""
    return sorted(
        [
            record
            for record in records
            if record["preconditioner"] == "fft_diffusion" and record["status"] == "completed"
        ],
        key=lambda record: int(record["trajectory_step"]),
    )


def _series(records: list[dict[str, Any]], name: str) -> list[dict[str, float]]:
    """Return one scalar diagnostic field against physical trajectory time."""
    return [{"time": float(record["time"]), name: float(record[name])} for record in records]


def _regime_summary(records: list[dict[str, Any]], regime: BrusselatorRegime) -> dict[str, Any]:
    """Summarize the evolved FFT-preconditioned states for one regime."""
    fft = _fft_records(records)
    disk_rates = [float(record["disk_rate"]) for record in fft]
    imag_extents = [float(record["fov_imaginary_extent"]) for record in fft]
    gmres_iterations = [float(record["actual_gmres"]["iterations"]) for record in fft]
    origin_flags = [bool(record["origin_enclosed"]) for record in fft]
    return {
        "parameters": regime._asdict(),
        "fft_records": len(fft),
        "verdict_distribution": _distribution(fft),
        "median_disk_rate": float(median(disk_rates)) if disk_rates else None,
        "median_fov_imaginary_extent": float(median(imag_extents)) if imag_extents else None,
        "origin_enclosed_fraction": (
            float(sum(origin_flags) / len(origin_flags)) if origin_flags else None
        ),
        "median_actual_fft_gmres_iterations": (
            float(median(gmres_iterations)) if gmres_iterations else None
        ),
        "fov_imaginary_extent_by_time": _series(fft, "fov_imaginary_extent"),
        "fft_gmres_iterations_by_time": [
            {
                "time": float(record["time"]),
                "iterations": int(record["actual_gmres"]["iterations"]),
                "converged": bool(record["actual_gmres"]["converged"]),
            }
            for record in fft
        ],
        "developedness_by_time": [
            {"time": float(record["time"]), **record["developedness"]} for record in fft
        ],
    }


def _hopf_vs_turing(
    comparison: dict[str, dict[str, Any]],
    config: DevelopedBrusselatorStudyConfig,
) -> dict[str, Any]:
    """Make a scale-qualified conclusion from developed-state measurements."""
    hopf = comparison["hopf"]
    turing = comparison["turing"]
    hopf_verdicts = hopf["verdict_distribution"]
    turing_verdicts = turing["verdict_distribution"]
    hopf_nonadequate = hopf_verdicts["investigate"] + hopf_verdicts["indeterminate"]
    turing_nonadequate = turing_verdicts["investigate"] + turing_verdicts["indeterminate"]
    hopf_origin_any = bool(
        hopf["origin_enclosed_fraction"] and hopf["origin_enclosed_fraction"] > 0.0
    )
    turing_origin_any = bool(
        turing["origin_enclosed_fraction"] and turing["origin_enclosed_fraction"] > 0.0
    )
    both_adequate = all(
        summary["verdict_distribution"]["adequate"] == summary["fft_records"]
        for summary in (hopf, turing)
    )
    both_indeterminate = all(
        summary["verdict_distribution"]["indeterminate"] == summary["fft_records"]
        for summary in (hopf, turing)
    )
    if both_adequate:
        outcome = "fft_adequate_throughout_developed_states"
        statement = (
            "The FFT diffusion preconditioner remains adequate at every sampled developed state "
            "in both regimes; no sampled Hopf field of values encloses the origin."
        )
    elif both_indeterminate:
        outcome = "both_regimes_indeterminate_on_developed_states"
        statement = (
            "Both evolved regimes are indeterminate at every sampled FFT-preconditioned state "
            "because their numerical ranges enclose the origin; Hopf still has the larger, "
            "growing imaginary extent."
        )
    elif hopf_nonadequate > turing_nonadequate or (hopf_origin_any and not turing_origin_any):
        outcome = "developed_hopf_coarsens"
        statement = (
            "Developed Hopf states have more investigate/indeterminate verdicts than Turing, "
            "or their field of values reaches the origin."
        )
    else:
        outcome = "other_developed_state_outcome"
        statement = (
            "The developed-state records do not support an all-adequate or Hopf-specific "
            "coarsening conclusion."
        )
    hopf_imaginary = hopf["fov_imaginary_extent_by_time"]
    hopf_imaginary_grows = (
        len(hopf_imaginary) > 1
        and hopf_imaginary[-1]["fov_imaginary_extent"] > hopf_imaginary[0]["fov_imaginary_extent"]
    )
    return {
        "outcome": outcome,
        "statement": statement,
        "hopf_nonadequate_fft_records": hopf_nonadequate,
        "turing_nonadequate_fft_records": turing_nonadequate,
        "hopf_origin_enclosed_any": hopf_origin_any,
        "turing_origin_enclosed_any": turing_origin_any,
        "both_regimes_indeterminate": both_indeterminate,
        "hopf_fov_imaginary_extent_grows_over_samples": hopf_imaginary_grows,
        "hopf_fov_imaginary_extent_by_time": hopf_imaginary,
        "scope_caveat": (
            f"This is a {config.nx}x{config.ny} screen with BE dt={config.dt:g}; "
            f"Hopf reaches t={config.dt * config.hopf_sample_steps[-1]:g} and Turing reaches "
            f"t={config.dt * config.turing_sample_steps[-1]:g}, below the paper's 256x256 scale. "
            "The FOV values use the dt=1 BE operator and therefore are not numerically "
            "interchangeable with the earlier dt=0.1 transient screen."
        ),
    }


def run_developed_brusselator_conditioning_study(
    config: DevelopedBrusselatorStudyConfig | None = None,
) -> dict[str, Any]:
    """Run the two-regime decision procedure on developed, not initial, states."""
    if config is None:
        config = DevelopedBrusselatorStudyConfig()
    if config.nx < 2 or config.ny < 2:
        raise ValueError("nx and ny must both be at least two")

    nk_params = NKParams(
        max_newton_iters=config.max_newton_iters,
        max_krylov_iters=config.max_krylov_iters,
        newton_tol=config.newton_tol,
        krylov_tol=config.krylov_tol,
    )
    records: list[dict[str, Any]] = []
    with jax.enable_x64(True):
        for regime_index, regime in enumerate((HOPF_REGIME, TURING_REGIME)):
            grid = Grid2D.uniform(
                config.nx,
                config.ny,
                0.0,
                regime.domain_length,
                0.0,
                regime.domain_length,
                n_ghost=1,
            )
            sample_steps = (
                config.hopf_sample_steps if regime.name == "hopf" else config.turing_sample_steps
            )
            samples = sampled_visited_states(
                regime,
                grid=grid,
                sample_steps=sample_steps,
                dt=config.dt,
                perturbation=config.perturbation,
                seed=config.seed + regime_index,
                nk_params=nk_params,
            )
            model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
            for sample in samples:
                for preconditioner_kind in ("identity", "fft_diffusion"):
                    assessment = assess_brusselator_state(
                        sample.state,
                        model,
                        fft_cache,
                        diffusivities,
                        config.dt,
                        regime,
                        preconditioner_kind=preconditioner_kind,
                        time_value=sample.time,
                        n_angles=config.n_angles,
                        fov_max_iters=config.fov_max_iters,
                        arnoldi_steps=config.arnoldi_steps,
                        seed=config.seed + 100 * regime_index + 10 * sample.step,
                    )
                    gmres = (
                        measure_brusselator_gmres(
                            sample.state,
                            model,
                            fft_cache,
                            diffusivities,
                            config.dt,
                            regime,
                            tol=config.krylov_tol,
                            max_iters=config.max_krylov_iters,
                            time_value=sample.time,
                            preconditioner_kind=preconditioner_kind,
                        )
                        if assessment["status"] == "completed"
                        else None
                    )
                    records.append(
                        {
                            **assessment,
                            "trajectory_step": sample.step,
                            "time": float(sample.time),
                            "developedness": sample.developedness,
                            "actual_gmres": gmres,
                        }
                    )

    by_regime = {
        regime.name: [record for record in records if record["regime"] == regime.name]
        for regime in (HOPF_REGIME, TURING_REGIME)
    }
    comparison = {
        regime.name: _regime_summary(by_regime[regime.name], regime)
        for regime in (HOPF_REGIME, TURING_REGIME)
    }
    return {
        "schema_version": "brusselator_conditioning_developed_v1",
        "status": (
            "completed"
            if all(record["status"] == "completed" for record in records)
            else "completed_with_skips"
        ),
        "config": config._asdict(),
        "model": {
            "name": "brusselator",
            "grid": [config.nx, config.ny],
            "boundary": "periodic",
            "spatial_operator": "moljax shipped periodic FFT-preconditioned path",
            "state_generation": "backward_euler_newton_krylov",
            "diagnostic_preconditioners": ["identity", "fft_diffusion"],
            "exact_solution_error": "not applicable: conditioning study",
        },
        "records": records,
        "regime_comparison": comparison,
        "hopf_vs_turing": _hopf_vs_turing(comparison, config),
    }


def main() -> None:
    """Run the developed-state study and write its reproducible JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/brusselator_conditioning_developed.json"),
    )
    args = parser.parse_args()
    result = run_developed_brusselator_conditioning_study(
        DevelopedBrusselatorStudyConfig(nx=args.nx, ny=args.ny, dt=args.dt)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
