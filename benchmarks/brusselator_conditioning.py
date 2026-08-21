#!/usr/bin/env python3
"""Two-regime Brusselator conditioning study on moljax's periodic FFT path.

The study evaluates genuinely visited backward-Euler states rather than the
homogeneous fixed point.  It compares the identity baseline with the shipped
FFT diffusion preconditioner and records both geometric diagnostics and a
matrix-free counted-GMRES ground truth.
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
    _integrate_visited_states,
    assess_brusselator_state,
    build_brusselator_system,
    measure_brusselator_gmres,
)


class BrusselatorStudyConfig(NamedTuple):
    """Configuration for a tractable 64² visited-state conditioning study.

    ``nx=ny=64`` is deliberately smaller than the paper's 256² target so the
    full matrix-free diagnostics can run at study cadence.  The Turing
    regime retains the paper's ``L=5`` and reference horizon ``t=200`` in its
    parameter record; this initial screen diagnoses the first few converged
    states, selected by ``n_states`` and ``dt``.
    """

    nx: int = 64
    ny: int = 64
    n_states: int = 2
    dt: float = 0.1
    perturbation: float = 1.0e-3
    seed: int = 20260821
    n_angles: int = 4
    fov_max_iters: int = 8
    arnoldi_steps: int = 6
    max_newton_iters: int = 10
    max_krylov_iters: int = 80
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    output_path: str = "benchmarks/results/brusselator_conditioning.json"


def _distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return all relevant verdict counts, including an explicit skipped bucket."""
    verdicts = {"adequate": 0, "investigate": 0, "indeterminate": 0, "skipped": 0}
    for record in records:
        verdicts[record["verdict"]] = verdicts.get(record["verdict"], 0) + 1
    return verdicts


def _finite_values(records: list[dict[str, Any]], name: str) -> list[float]:
    """Extract finite completed-record values from a JSON-ready record list."""
    return [float(record[name]) for record in records if record["status"] == "completed"]


def _regime_summary(records: list[dict[str, Any]], regime: BrusselatorRegime) -> dict[str, Any]:
    """Summarize FFT-preconditioned diagnostics for one physical regime."""
    fft_records = [record for record in records if record["preconditioner"] == "fft_diffusion"]
    disk_rates = _finite_values(fft_records, "disk_rate")
    imag_extents = _finite_values(fft_records, "fov_imaginary_extent")
    origin_flags = [
        bool(record["origin_enclosed"]) for record in fft_records if record["status"] == "completed"
    ]
    fft_iterations = [
        float(record["actual_gmres"]["iterations"])
        for record in fft_records
        if record["actual_gmres"] is not None
    ]
    return {
        "parameters": regime._asdict(),
        "fft_records": len(fft_records),
        "verdict_distribution": _distribution(fft_records),
        "median_disk_rate": float(median(disk_rates)) if disk_rates else None,
        "median_fov_imaginary_extent": float(median(imag_extents)) if imag_extents else None,
        "origin_enclosed_fraction": (
            float(sum(origin_flags) / len(origin_flags)) if origin_flags else None
        ),
        "median_actual_fft_gmres_iterations": (
            float(median(fft_iterations)) if fft_iterations else None
        ),
    }


def _hopf_vs_turing(comparison: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """State the two-regime outcome only from the collected summary statistics."""
    hopf = comparison["hopf"]
    turing = comparison["turing"]
    hopf_nonadequate = (
        hopf["verdict_distribution"]["investigate"] + hopf["verdict_distribution"]["indeterminate"]
    )
    turing_nonadequate = (
        turing["verdict_distribution"]["investigate"]
        + turing["verdict_distribution"]["indeterminate"]
    )
    nonadequate_separation = hopf_nonadequate > turing_nonadequate or (
        hopf["origin_enclosed_fraction"] is not None
        and turing["origin_enclosed_fraction"] is not None
        and hopf["origin_enclosed_fraction"] > turing["origin_enclosed_fraction"]
    )
    larger_hopf_imaginary_extent = (
        hopf["median_fov_imaginary_extent"] is not None
        and turing["median_fov_imaginary_extent"] is not None
        and hopf["median_fov_imaginary_extent"] > turing["median_fov_imaginary_extent"]
    )
    both_adequate = all(
        summary["verdict_distribution"]["adequate"] == summary["fft_records"]
        for summary in (hopf, turing)
    )
    if both_adequate:
        outcome = "both_adequate_under_fft"
        statement = (
            "The FFT diffusion preconditioner is assessed adequate for both visited-state regimes; "
            "the larger Hopf imaginary extent is recorded, but it does not change the decision verdict."
        )
    elif nonadequate_separation:
        outcome = "distinguishes_regimes"
        statement = (
            "Hopf has more non-adequate geometry, a larger origin-enclosed fraction, "
            "than Turing in these visited states."
        )
    else:
        outcome = "no_clear_separation"
        statement = (
            "The collected geometric diagnostics do not establish a clear Hopf-versus-Turing "
            "separation at this study cadence."
        )
    return {
        "outcome": outcome,
        "statement": statement,
        "hopf_nonadequate_fft_records": hopf_nonadequate,
        "turing_nonadequate_fft_records": turing_nonadequate,
        "hopf_median_fov_imaginary_extent": hopf["median_fov_imaginary_extent"],
        "turing_median_fov_imaginary_extent": turing["median_fov_imaginary_extent"],
        "hopf_origin_enclosed_fraction": hopf["origin_enclosed_fraction"],
        "turing_origin_enclosed_fraction": turing["origin_enclosed_fraction"],
        "nonadequate_separation": nonadequate_separation,
        "larger_hopf_imaginary_extent": larger_hopf_imaginary_extent,
        "both_adequate": both_adequate,
    }


def run_brusselator_conditioning_study(
    config: BrusselatorStudyConfig | None = None,
) -> dict[str, Any]:
    """Run the two-regime visited-state FFT-conditioning study.

    No exact-solution error is reported because this is a conditioning study.
    Each record instead contains an adjoint gate, field-of-values geometry,
    and a counted matrix-free GMRES solve for the identical preconditioned
    linear system.
    """
    if config is None:
        config = BrusselatorStudyConfig()
    if config.nx < 2 or config.ny < 2:
        raise ValueError("nx and ny must both be at least two")
    if config.n_states < 1:
        raise ValueError("n_states must be positive")

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
            model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
            states = _integrate_visited_states(
                regime,
                model,
                fft_cache,
                n_steps=config.n_states,
                dt=config.dt,
                perturbation=config.perturbation,
                seed=config.seed + regime_index,
                nk_params=nk_params,
            )
            for state_index, state in enumerate(states):
                time_value = (state_index + 1) * config.dt
                for preconditioner_kind in ("identity", "fft_diffusion"):
                    assessment = assess_brusselator_state(
                        state,
                        model,
                        fft_cache,
                        diffusivities,
                        config.dt,
                        regime,
                        preconditioner_kind=preconditioner_kind,
                        time_value=time_value,
                        n_angles=config.n_angles,
                        fov_max_iters=config.fov_max_iters,
                        arnoldi_steps=config.arnoldi_steps,
                        seed=config.seed + 100 * regime_index + 10 * state_index,
                    )
                    if assessment["status"] == "completed":
                        gmres = measure_brusselator_gmres(
                            state,
                            model,
                            fft_cache,
                            diffusivities,
                            config.dt,
                            regime,
                            tol=config.krylov_tol,
                            max_iters=config.max_krylov_iters,
                            time_value=time_value,
                            preconditioner_kind=preconditioner_kind,
                        )
                    else:
                        gmres = None
                    records.append(
                        {
                            **assessment,
                            "state_index": state_index,
                            "time": float(time_value),
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
        "schema_version": "brusselator_conditioning_v1",
        "status": (
            "completed"
            if all(row["status"] == "completed" for row in records)
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
        "hopf_vs_turing": _hopf_vs_turing(comparison),
    }


def main() -> None:
    """Run the default study and write JSON-ready results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/brusselator_conditioning.json"),
    )
    args = parser.parse_args()
    result = run_brusselator_conditioning_study(
        BrusselatorStudyConfig(nx=args.nx, ny=args.ny, n_states=args.n_states, dt=args.dt)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
