"""Smoke tests for the experimental two-regime Brusselator conditioning study."""

from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import pytest

from benchmarks import brusselator_conditioning as benchmark
from benchmarks import resolve_brusselator_fov_supports as resolver
from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    assess_brusselator_state,
    build_brusselator_system,
    sampled_visited_states,
    visited_states,
)
from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
    dense_padded_preconditioned_operator,
)


def _homogeneous_state(regime, grid):
    model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
    state = model.apply_bcs(
        {
            "u": jax.numpy.full(
                (grid.ny_total, grid.nx_total), regime.a, dtype=jax.numpy.float64
            ),
            "v": jax.numpy.full(
                (grid.ny_total, grid.nx_total), regime.b / regime.a, dtype=jax.numpy.float64
            ),
        },
        0.0,
    )
    return state, model, fft_cache, diffusivities


@pytest.fixture(scope="module")
def tiny_fft_records():
    """Evaluate the minimal FFT-only two-regime smoke configuration once."""
    records = {}
    for seed, regime in enumerate((HOPF_REGIME, TURING_REGIME), start=1):
        grid = Grid2D.uniform(8, 8, 0.0, regime.domain_length, 0.0, regime.domain_length)
        state = visited_states(
            regime,
            grid=grid,
            n_steps=1,
            dt=0.1,
            perturbation=1.0e-3,
            seed=seed,
        )[0]
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        records[regime.name] = assess_brusselator_state(
            state,
            model,
            fft_cache,
            diffusivities,
            0.1,
            regime,
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=3,
            seed=seed,
        )
    return records


@pytest.mark.slow
def test_hopf_visited_state_passes_adjoint_gate_and_has_a_verdict(tiny_fft_records):
    """A tiny FFT-preconditioned Hopf state is valid input to the toolbox."""
    record = tiny_fft_records["hopf"]
    assert record["status"] == "completed"
    assert record["adjoint_error"] <= 1.0e-8
    assert record["verdict"] in {"adequate", "investigate", "indeterminate"}


@pytest.mark.slow
def test_both_regimes_record_structural_discriminators(tiny_fft_records):
    """The outcome is data, but both physical-regime record fields must exist."""
    for regime in ("hopf", "turing"):
        record = tiny_fft_records[regime]
        assert record["status"] == "completed"
        assert record["adjoint_error"] <= 1.0e-8
        assert isinstance(record["origin_enclosed"], bool)
        assert math.isfinite(record["fov_imaginary_extent"])
        assert record["fov_imaginary_extent"] >= 0.0


@pytest.mark.slow
def test_developed_hopf_sample_leaves_the_fixed_point_and_passes_adjoint_gate():
    """A late sampled Hopf state is developed rather than a seed perturbation."""
    perturbation = 1.0e-3
    grid = Grid2D.uniform(
        8,
        8,
        0.0,
        HOPF_REGIME.domain_length,
        0.0,
        HOPF_REGIME.domain_length,
    )
    samples = sampled_visited_states(
        HOPF_REGIME,
        grid=grid,
        sample_steps=(1, 5, 10),
        dt=1.0,
        perturbation=perturbation,
        seed=20260822,
        nk_params=NKParams(
            max_newton_iters=15,
            max_krylov_iters=100,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        ),
    )
    late = samples[-1]
    model, fft_cache, diffusivities = build_brusselator_system(HOPF_REGIME, grid)
    assessment = assess_brusselator_state(
        late.state,
        model,
        fft_cache,
        diffusivities,
        1.0,
        HOPF_REGIME,
        n_angles=3,
        fov_max_iters=4,
        arnoldi_steps=3,
        seed=20260822,
    )

    departure = max(late.developedness.values())
    assert departure > 20.0 * perturbation
    assert assessment["status"] == "completed"
    assert assessment["adjoint_error"] <= 1.0e-8


@pytest.mark.slow
def test_fourier_weyl_bound_certifies_a_small_homogeneous_turing_state():
    """The integrated full-operator bound unlocks adequacy only with all gates clear."""
    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.01,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        compute_lobpcg_upper_estimate=True,
        seed=20260821,
    )
    certificate = assessment["fourier_weyl_ghost_lower_bound"]
    interior = np.asarray(state["u"])[1:-1, 1:-1]
    interior_v = np.asarray(state["v"])[1:-1, 1:-1]
    dense = float(
        np.linalg.svd(
            dense_padded_preconditioned_operator(
                interior,
                interior_v,
                du=TURING_REGIME.du,
                dv=TURING_REGIME.dv,
                beta=TURING_REGIME.b,
                dt=0.01,
            ),
            compute_uv=False,
        )[-1]
    )
    assert assessment["verdict"] == "adequate"
    assert assessment["epsilon_zero_full_operator_evidence"] is True
    assert certificate["status"] == "clears_adequacy_gate"
    assert certificate["full_lower_bound"] >= 0.1
    assert certificate["full_lower_bound"] <= dense + 5.0e-13
    assert assessment["lobpcg_sigma_min_upper_estimate"] is not None


@pytest.mark.slow
@pytest.mark.parametrize("weak_bound", [0.0, 0.05])
def test_valid_weak_fourier_weyl_bound_preserves_provisional(monkeypatch, weak_bound):
    """A valid but insufficient lower bound cannot promote a provisional reading."""
    import moljax.experimental.brusselator_conditioning as conditioning

    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    genuine = conditioning._fourier_weyl_bound(state, model, TURING_REGIME, 0.01)
    weak_selected = replace(genuine.selected, full_lower_bound=weak_bound)
    monkeypatch.setattr(
        conditioning,
        "_fourier_weyl_bound",
        lambda *_args, **_kwargs: replace(genuine, selected=weak_selected),
    )
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.01,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        seed=20260821,
    )
    certificate = assessment["fourier_weyl_ghost_lower_bound"]
    assert genuine.selected.full_lower_bound >= 0.1
    assert assessment["verdict"] == "provisional"
    assert assessment["epsilon_zero_full_operator_evidence"] is False
    assert certificate["status"] == "valid_but_below_adequacy_gate"
    assert certificate["full_lower_bound"] == pytest.approx(weak_bound)


@pytest.mark.slow
def test_origin_enclosure_remains_indeterminate_despite_a_valid_bound():
    """The full-operator lower bound cannot override an origin-enclosed FOV."""
    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state = visited_states(
        TURING_REGIME,
        grid=grid,
        n_steps=1,
        dt=0.2,
        perturbation=0.8,
        seed=20260928,
    )[0]
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.2,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        seed=20260928,
    )
    assert assessment["fourier_weyl_ghost_lower_bound"]["full_lower_bound"] >= 0.0
    assert assessment["origin_enclosed"] is True
    assert assessment["verdict"] == "indeterminate"


def test_v4_source_cache_rejects_a_foreign_generation_fingerprint(tmp_path):
    """A foreign source-generation contract is a safe cache miss."""
    config = benchmark._config(
        "screen_64",
        nx=4,
        ny=4,
        n_states=1,
        source_state_cache_dir=str(tmp_path),
    )
    fingerprint = benchmark._source_state_fingerprint(config, TURING_REGIME, (1,), 7)
    state = {
        "u": jax.numpy.ones((6, 6), dtype=jax.numpy.float64),
        "v": jax.numpy.full((6, 6), 1.8, dtype=jax.numpy.float64),
    }
    benchmark._persist_source_states(config, TURING_REGIME, fingerprint, [state])
    foreign = {**fingerprint, "seed": 8}
    assert benchmark._load_cached_states(config, TURING_REGIME, foreign) is None


def _replay_fixture(tmp_path):
    """Persist one minimal source artifact and return its exact replay record."""
    cache_root = tmp_path / "original-cache"
    config = benchmark._config(
        "screen_64",
        nx=4,
        ny=4,
        n_states=1,
        source_state_cache_dir=str(cache_root),
    )
    fingerprint = benchmark._source_state_fingerprint(config, TURING_REGIME, (1,), 7)
    state = {
        "u": jax.numpy.ones((6, 6), dtype=jax.numpy.float64),
        "v": jax.numpy.full((6, 6), 1.8, dtype=jax.numpy.float64),
    }
    _, identities = benchmark._persist_source_states(config, TURING_REGIME, fingerprint, [state])
    array_path, _ = benchmark._cache_paths(config, TURING_REGIME, fingerprint)
    record = {
        "time": config.dt,
        "record_config": {
            "regime": TURING_REGIME._asdict(),
            "grid": {"nx": 4, "ny": 4, "n_ghost": 1},
            "source_state_fingerprint": fingerprint,
            "analysis_dt": config.dt,
            "preconditioner_kind": "identity",
            "n_angles": config.n_angles,
            "fov_max_iters": config.fov_max_iters,
            "fov_residual_tolerance": config.fov_residual_tolerance,
            "fov_n_restarts": config.fov_n_restarts,
            "arnoldi_steps": config.arnoldi_steps,
            "compute_lobpcg_upper_estimate": config.compute_lobpcg_upper_estimate,
            "assessment_seed": config.seed,
            "domain_length": TURING_REGIME.domain_length,
        },
        "source_state_artifact": {
            "schema": benchmark.SOURCE_STATE_ARTIFACT_SCHEMA,
            "relative_path": array_path.name,
            "sample_position": 0,
            "source_state_identity": identities[0],
            "generation_fingerprint": fingerprint,
            "converged": True,
        },
    }
    return cache_root, record, state


def test_v4_source_cache_replay_survives_relocation(tmp_path, monkeypatch):
    """A cache-root-relative artifact replays after an intact cache is moved."""
    cache_root, record, original_state = _replay_fixture(tmp_path)
    relocated = tmp_path / "relocated-cache"
    shutil.move(str(cache_root), relocated)
    expected_identity = benchmark._source_state_identity(original_state)
    monkeypatch.setattr(
        benchmark,
        "assess_brusselator_state",
        lambda state, *_args, **_kwargs: {
            "loaded_source_identity": benchmark._source_state_identity(state)
        },
    )

    replayed = benchmark.reassess_brusselator_record(
        record, source_state_cache_dir=str(relocated)
    )

    assert replayed["loaded_source_identity"] == expected_identity


@pytest.mark.parametrize("bad_path", ["/tmp/foreign-state.npz", "../foreign-state.npz"])
def test_v4_source_cache_replay_rejects_nonrelative_artifact_paths(tmp_path, bad_path):
    """Relocatable provenance never permits an absolute or traversing path."""
    cache_root, record, _ = _replay_fixture(tmp_path)
    record["source_state_artifact"]["relative_path"] = bad_path

    with pytest.raises(RuntimeError, match="cache-root-relative"):
        benchmark.reassess_brusselator_record(record, source_state_cache_dir=str(cache_root))


def _short_arnoldi_counterexample():
    """Return Pavlov's deterministic incomplete-reading counterexample."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        u = 1.0 + rng.uniform(-0.8, 0.8, (4, 4))
        v = 1.8 + rng.uniform(-1.0, 1.0, (4, 4))
    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    state = model.apply_bcs(
        {
            "u": jax.numpy.zeros((6, 6), dtype=jax.numpy.float64)
            .at[1:-1, 1:-1]
            .set(jax.numpy.asarray(u)),
            "v": jax.numpy.zeros((6, 6), dtype=jax.numpy.float64)
            .at[1:-1, 1:-1]
            .set(jax.numpy.asarray(v)),
        },
        0.0,
    )
    return assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.2,
        TURING_REGIME,
        n_angles=4,
        fov_max_iters=60,
        fov_n_restarts=2,
        arnoldi_steps=1,
        seed=0,
    )


@pytest.mark.slow
def test_weak_bound_never_overrides_a_short_arnoldi_abstention():
    """Incomplete Ritz evidence remains indeterminate in module and resolver policy."""
    assessment = _short_arnoldi_counterexample()

    assert assessment["fourier_weyl_ghost_lower_bound"]["full_lower_bound"] < 0.1
    assert assessment["n_right_real_outliers"] is None
    assert assessment["verdict"] == "indeterminate"
    assert resolver._policy_category(assessment) == "indeterminate"


def test_weak_bound_never_overrides_a_nonfinite_reading():
    """A non-finite/invalid reading remains an abstention in both policy sites."""
    invalid = {
        "verdict": "indeterminate",
        "verdict_reason": "ritz contains a non-finite value",
        "disk_rate": float("nan"),
        "epsilon_zero": float("nan"),
        "n_right_real_outliers": None,
        "supports_consistent": True,
        "origin_enclosed": False,
        "fourier_weyl_ghost_lower_bound": {"status": "valid_but_below_adequacy_gate"},
    }
    module_assessment = SimpleNamespace(
        verdict="indeterminate", n_right_real_outliers=None
    )

    assert resolver._policy_category(invalid) == "indeterminate"
    assert resolver._policy_outcome(invalid) == ("indeterminate", invalid["verdict_reason"])
    from moljax.experimental import brusselator_conditioning as conditioning

    assert conditioning._weak_bound_override_eligible(module_assessment) is False


def _resolved_reports() -> dict[str, dict]:
    """Load the four promoted final-policy reports committed with the study."""
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
    names = {
        "screen_64": "brusselator_conditioning.json",
        "developed_64": "brusselator_conditioning_developed.json",
        "fixed_dt_256": "brusselator_conditioning_fixed_dt.json",
        "hopf_continuation_256": "brusselator_conditioning_hopf_continuation.json",
    }
    import json

    return {study: json.loads((root / name).read_text()) for study, name in names.items()}


def test_promoted_resolved_reports_match_final_policy_and_tally():
    """Every published record and the aggregate tally use the terminal policy."""
    reports = _resolved_reports()
    tally = {
        "adequate": 0,
        "provisional": 0,
        "investigate": 0,
        "indeterminate": 0,
        "uncertified_at_cap": 0,
    }
    for report in reports.values():
        for record in report["records"]:
            assert record["verdict"] == record["final_verdict"] == record["final_category"]
            assert not Path(record["source_state_artifact"]["relative_path"]).is_absolute()
            assert ".." not in Path(record["source_state_artifact"]["relative_path"]).parts
            tally[record["verdict"]] += 1
    assert tally == {
        "adequate": 13,
        "provisional": 1,
        "investigate": 5,
        "indeterminate": 12,
        "uncertified_at_cap": 1,
    }


def test_promoted_resolved_report_aggregates_match_final_records():
    """All record-derived summaries are recomputed after FOV resolution."""
    reports = _resolved_reports()
    for study, report in reports.items():
        rebuilt = dict(report)
        resolver._recompute_derived_summaries(rebuilt, report["records"], study)
        for key in ("regime_comparison", "hopf_vs_turing", "fixed_dt_transition"):
            if key in report:
                assert report[key] == rebuilt[key]

    screen = reports["screen_64"]
    assert screen["regime_comparison"]["hopf"]["verdict_distribution"]["adequate"] == 2
    assert screen["regime_comparison"]["turing"]["verdict_distribution"]["adequate"] == 2
    assert screen["regime_comparison"]["hopf"]["median_disk_rate"] == pytest.approx(
        0.4404371091303861
    )
    assert screen["hopf_vs_turing"]["hopf_adequate_fft_records"] == 2
    assert screen["hopf_vs_turing"]["turing_adequate_fft_records"] == 2

    fixed = reports["fixed_dt_256"]
    for regime in ("hopf", "turing"):
        for kind in ("identity", "fft_diffusion"):
            rows = sorted(
                (
                    record
                    for record in fixed["records"]
                    if record["regime"] == regime and record["preconditioner"] == kind
                ),
                key=lambda record: record["trajectory_step"],
            )
            transition = fixed["fixed_dt_transition"]["by_regime"][regime][kind]
            assert transition["early"]["verdict"] == rows[0]["verdict"]
            assert transition["developed"]["verdict"] == rows[-1]["verdict"]
