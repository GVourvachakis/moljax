#!/usr/bin/env python3
"""Fixed-step 256² Brusselator conditioning comparison at paper resolution.

This experiment holds the backward-Euler timestep fixed while comparing an
early perturbed state with a developed state from the same periodic
Brusselator trajectory.  It therefore avoids attributing a state-evolution
change in field-of-values geometry to a changed timestep.  The linearized
Jacobian still changes with the visited state, as it should for this nonlinear
conditioning study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax

from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    assess_brusselator_state,
    build_brusselator_system,
    measure_brusselator_gmres,
    sampled_visited_states,
)


class PaperResolutionBrusselatorStudyConfig(NamedTuple):
    """A fixed-step early-versus-developed study at the paper's 256² grid.

    ``dt=0.2`` was selected by a 256² preflight: both one-step FFT states
    remain adequate, whereas ``dt=0.5`` already encloses the origin.  Hopf is
    sampled at ``t=10`` after the oscillation has left the seed perturbation;
    Turing reaches its paper horizon ``t=200``.
    """

    nx: int = 256
    ny: int = 256
    dt: float = 0.2
    perturbation: float = 1.0e-3
    hopf_sample_steps: tuple[int, int] = (1, 50)
    turing_sample_steps: tuple[int, int] = (1, 1000)
    seed: int = 20260823
    n_angles: int = 4
    fov_max_iters: int = 8
    arnoldi_steps: int = 6
    max_newton_iters: int = 15
    max_krylov_iters: int = 100
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    output_path: str = "benchmarks/results/brusselator_conditioning_paperres.json"


def _phase_records(
    records: list[dict[str, Any]],
    regime: str,
    preconditioner: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the early and developed records for one regime/preconditioner."""
    selected = sorted(
        [
            record
            for record in records
            if record["regime"] == regime and record["preconditioner"] == preconditioner
        ],
        key=lambda record: int(record["trajectory_step"]),
    )
    if len(selected) != 2:
        raise ValueError(f"Expected exactly two {regime}/{preconditioner} records")
    return selected[0], selected[1]


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON fields relevant to the fixed-step state comparison."""
    gmres = record["actual_gmres"]
    return {
        "trajectory_step": int(record["trajectory_step"]),
        "time": float(record["time"]),
        "developedness": record["developedness"],
        "verdict": record["verdict"],
        "disk_rate": record["disk_rate"],
        "epsilon_zero": record["epsilon_zero"],
        "origin_enclosed": record["origin_enclosed"],
        "fov_imaginary_extent": record["fov_imaginary_extent"],
        "n_right_real_outliers": record["n_right_real_outliers"],
        "adjoint_error": record["adjoint_error"],
        "actual_gmres_iterations": None if gmres is None else int(gmres["iterations"]),
        "actual_gmres_converged": None if gmres is None else bool(gmres["converged"]),
        "actual_gmres_final_relative_residual": (
            None if gmres is None else float(gmres["final_relative_residual"])
        ),
    }


def _transition_for_regime(records: list[dict[str, Any]], regime: str) -> dict[str, Any]:
    """Summarize each preconditioner's fixed-timestep state evolution."""
    comparison: dict[str, Any] = {}
    for preconditioner in ("identity", "fft_diffusion"):
        early, developed = _phase_records(records, regime, preconditioner)
        early_summary = _record_summary(early)
        developed_summary = _record_summary(developed)
        comparison[preconditioner] = {
            "early": early_summary,
            "developed": developed_summary,
            "adequate_to_indeterminate": (
                early_summary["verdict"] == "adequate"
                and developed_summary["verdict"] == "indeterminate"
            ),
        }
    return comparison


def _fixed_dt_transition(
    records: list[dict[str, Any]], config: PaperResolutionBrusselatorStudyConfig
) -> dict[str, Any]:
    """Describe the data-derived early/developed conclusion at one timestep."""
    by_regime = {
        regime.name: _transition_for_regime(records, regime.name)
        for regime in (HOPF_REGIME, TURING_REGIME)
    }
    hopf_fft = by_regime["hopf"]["fft_diffusion"]
    turing_fft = by_regime["turing"]["fft_diffusion"]
    both_transition = (
        hopf_fft["adequate_to_indeterminate"] and turing_fft["adequate_to_indeterminate"]
    )
    any_transition = (
        hopf_fft["adequate_to_indeterminate"] or turing_fft["adequate_to_indeterminate"]
    )
    if both_transition:
        outcome = "fft_adequate_to_indeterminate_in_both_regimes_at_fixed_dt"
        statement = (
            "At fixed backward-Euler dt, the FFT-preconditioned verdict changes from adequate "
            "at the early state to indeterminate at the developed state in both regimes."
        )
    elif any_transition:
        outcome = "fft_adequate_to_indeterminate_in_one_regime_at_fixed_dt"
        statement = (
            "At fixed backward-Euler dt, at least one FFT-preconditioned regime changes from "
            "adequate early to indeterminate after its state develops; see the per-regime rows."
        )
    else:
        outcome = "no_fft_adequate_to_indeterminate_transition_at_fixed_dt"
        statement = (
            "At this fixed backward-Euler dt, the records do not show an FFT-preconditioned "
            "adequate-to-indeterminate transition in either regime."
        )
    return {
        "outcome": outcome,
        "statement": statement,
        "fixed_dt": config.dt,
        "same_discretized_operator_family": (
            "Every early/developed pair uses the same periodic 256x256 grid, shipped FFT "
            "preconditioner, and backward-Euler timestep.  The state-dependent Jacobian changes "
            "between visited states by design; no comparison changes dt."
        ),
        "by_regime": by_regime,
    }


def run_paper_resolution_brusselator_conditioning_study(
    config: PaperResolutionBrusselatorStudyConfig | None = None,
) -> dict[str, Any]:
    """Run fixed-dt diagnostics at 256² on early and developed states.

    The diagnostic budget is intentionally modest (four numerical-range
    angles, eight Hermitian power iterations, and six Arnoldi steps) so that
    the matrix-free calculation fits in the RTX 3090 Ti memory budget.  Both
    the identity baseline and the shipped FFT diffusion preconditioner use
    the same counted-GMRES tolerance.
    """
    if config is None:
        config = PaperResolutionBrusselatorStudyConfig()
    if config.nx < 2 or config.ny < 2:
        raise ValueError("nx and ny must both be at least two")
    if config.dt <= 0.0:
        raise ValueError("dt must be positive")

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

    transition = _fixed_dt_transition(records, config)
    return {
        "schema_version": "brusselator_conditioning_paperres_v1",
        "status": (
            "completed"
            if all(record["status"] == "completed" for record in records)
            else "completed_with_skips"
        ),
        "config": config._asdict(),
        "model": {
            "name": "brusselator",
            "grid": [config.nx, config.ny],
            "domain_length": HOPF_REGIME.domain_length,
            "boundary": "periodic",
            "spatial_operator": "moljax shipped periodic pseudo-spectral FFT path",
            "state_generation": "FFT-preconditioned backward_euler_newton_krylov",
            "diagnostic_preconditioners": ["identity", "fft_diffusion"],
            "exact_solution_error": "not applicable: conditioning study",
        },
        "records": records,
        "fixed_dt_transition": transition,
        "scope": {
            "paper_resolution": "256x256 physical periodic grid at L=5",
            "hopf_developed_time": config.dt * config.hopf_sample_steps[-1],
            "turing_developed_time": config.dt * config.turing_sample_steps[-1],
            "turing_reaches_paper_t200": (
                config.dt * config.turing_sample_steps[-1] == TURING_REGIME.reference_final_time
            ),
            "caveat": (
                "The Turing developed state reaches t=200.  The Hopf sample is a developed "
                "state at the stated time, not a claim to reproduce a full long-time attractor "
                "unless its recorded developedness supports that interpretation."
            ),
        },
    }


def main() -> None:
    """Run the paper-resolution fixed-dt comparison and write JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/brusselator_conditioning_paperres.json"),
    )
    args = parser.parse_args()
    result = run_paper_resolution_brusselator_conditioning_study(
        PaperResolutionBrusselatorStudyConfig(nx=args.nx, ny=args.ny, dt=args.dt)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
