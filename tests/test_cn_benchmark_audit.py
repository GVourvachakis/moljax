"""Static error matrices and bitwise fidelity to the paper's CN loops."""

from __future__ import annotations

import ast
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


def _original_cn_step(name: str, n: int):
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
    h = 0.5 * cfg["dt"]
    namespace = {
        "jax": jax,
        "jnp": jnp,
        "lap_eig_jax": lam,
        "cn_precond_u": 1.0 / (1.0 - h * cfg["du"] * lam),
        "cn_precond_v": 1.0 / (1.0 - h * cfg["dv"] * lam),
        "dt_cn": cfg["dt"],
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


@pytest.mark.parametrize("name", tuple(audit.CONFIGS))
def test_five_instrumented_steps_are_bitwise_original(name: str) -> None:
    original = _original_cn_step(name, 32)
    replica, _ = audit.make_step(name, 32, "fft")
    uninstrumented = tuple(jnp.asarray(x) for x in audit.initial_state(name, 32))
    instrumented = uninstrumented
    for _ in range(5):
        uninstrumented = original(*uninstrumented)
        instrumented, _ = replica(instrumented)
        for before, after in zip(uninstrumented, instrumented, strict=True):
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
