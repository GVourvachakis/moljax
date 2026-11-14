"""Static error matrices and bitwise fidelity to the paper's CN loops."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import cn_benchmark_audit as audit


def test_m0_closed_form_matches_dense_eigenvalues() -> None:
    blocks = np.array([[[4.0, 5.0], [-9.0, -5.0]], [[0.2, -1.0], [0.3, 0.4]]])
    expected = np.max(np.abs(np.linalg.eigvals(blocks)), axis=-1)
    actual = audit.rho_2x2(blocks[:, 0, 0], blocks[:, 0, 1], blocks[:, 1, 0], blocks[:, 1, 1])
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert actual[0] == 5.0


@pytest.mark.parametrize("name", tuple(audit.CONFIGS))
def test_mean_field_error_is_zero_at_homogeneous_state(name: str) -> None:
    result = audit.m0(name)
    assert result["mean_field_error_max_abs"] <= 1e-12
    assert result["mean_field_blocks_nonsingular"]


def _original_cn_step(name: str, n: int, dt_override: float | None = None):
    cfg = audit.CONFIGS[name]
    path = Path(audit.ROOT / "benchmarks" / cfg["script"])
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == cfg["function"]
    )
    source = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    lam = jnp.asarray(audit.lap_symbol(n))
    dt = cfg["dt"] if dt_override is None else dt_override
    h = 0.5 * dt
    namespace = {
        "jax": jax,
        "jnp": jnp,
        "lap_eig_jax": lam,
        "cn_precond_u": 1.0 / (1.0 - h * cfg["du"] * lam),
        "cn_precond_v": 1.0 / (1.0 - h * cfg["dv"] * lam),
        "dt_cn": dt,
        "Du": cfg["du"],
        "Dv": cfg["dv"],
        "CN_TOL": cfg["tol"],
        "CN_MAX_ITER": cfg["cap"],
        "a": cfg.get("a"),
        "b": cfg.get("b"),
        "gamma": cfg.get("gamma"),
        "F": cfg.get("feed"),
        "k": cfg.get("kill"),
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[cfg["function"]]


@pytest.mark.parametrize("name", ("brusselator", "gray_scott"))
def test_five_instrumented_steps_are_bitwise_original(name: str) -> None:
    original = _original_cn_step(name, 32)
    replica, _ = audit.make_step(name, 32, "fft")
    uninstrumented = tuple(jnp.asarray(x) for x in audit.initial_state(name, 32))
    instrumented = uninstrumented
    for _ in range(5):
        uninstrumented = original(*uninstrumented)
        instrumented, _ = replica(instrumented)
        for before, after in zip(uninstrumented, instrumented, strict=True):
            assert np.isfinite(np.asarray(before)).all()
            assert np.isfinite(np.asarray(after)).all()
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()


def test_schnakenberg_finite_data_fidelity_at_dt_1e_minus_3() -> None:
    original = _original_cn_step("schnakenberg", 32, dt_override=1e-3)
    replica, _ = audit.make_step("schnakenberg", 32, "fft", dt_override=1e-3)
    uninstrumented = tuple(jnp.asarray(x) for x in audit.initial_state("schnakenberg", 32))
    instrumented = uninstrumented
    for _ in range(5):
        uninstrumented = original(*uninstrumented)
        instrumented, _ = replica(instrumented)
        for before, after in zip(uninstrumented, instrumented, strict=True):
            assert np.isfinite(np.asarray(before)).all()
            assert np.isfinite(np.asarray(after)).all()
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()


def test_schnakenberg_paper_dt_has_matching_nonfinite_step() -> None:
    original = _original_cn_step("schnakenberg", 32)
    replica, _ = audit.make_step("schnakenberg", 32, "fft")
    uninstrumented = tuple(jnp.asarray(x) for x in audit.initial_state("schnakenberg", 32))
    instrumented = uninstrumented
    original_finite, replica_finite = [], []
    for _ in range(5):
        uninstrumented = original(*uninstrumented)
        instrumented, _ = replica(instrumented)
        original_finite.append(all(np.isfinite(np.asarray(x)).all() for x in uninstrumented))
        replica_finite.append(all(np.isfinite(np.asarray(x)).all() for x in instrumented))
    assert original_finite == replica_finite == [False] * 5


def _paper_initial_state(name: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = audit.CONFIGS[name]
    path = audit.ROOT / "benchmarks" / cfg["script"]
    tree = ast.parse(path.read_text())
    seed_index = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "np.random.seed"
    )
    end_index = next(
        i
        for i in range(seed_index + 1, len(tree.body))
        if isinstance(tree.body[i], ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "u0_jax"
            for target in tree.body[i].targets
        )
    )
    steady_state = [
        node
        for node in tree.body[:seed_index]
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"u_s", "v_s"} for target in node.targets
        )
    ]
    source = ast.fix_missing_locations(
        ast.Module(body=steady_state + tree.body[seed_index:end_index], type_ignores=[])
    )
    namespace = {"np": np, "N": cfg["n"], "a": cfg.get("a"), "b": cfg.get("b")}
    rng_state = np.random.get_state()
    try:
        exec(compile(source, str(path), "exec"), namespace)
    finally:
        np.random.set_state(rng_state)
    return namespace["u0"], namespace["v0"]


@pytest.mark.parametrize("name", tuple(audit.CONFIGS))
def test_paper_initial_state_is_bitwise_identical(name: str) -> None:
    cfg = audit.CONFIGS[name]
    expected_state = _paper_initial_state(name)
    actual = audit.initial_state(name, cfg["n"])
    for expected, computed in zip(expected_state, actual, strict=True):
        assert expected.tobytes() == computed.tobytes()


def test_post_run_report_matches_unchanged_measurements() -> None:
    results = audit.ROOT / "benchmarks/results"
    payload = json.loads((results / "cn_benchmark_audit.json").read_text())
    rendered = audit.render(payload)
    assert rendered == (results / "CN_BENCHMARK_AUDIT_RESULTS.md").read_text()
    assert "Added after the pre-registered run:" in rendered
