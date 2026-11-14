"""Audit the three paper CN loops without importing their executable scripts.

Run stages separately; all intermediate states and logs belong in --artifact-dir.
Only ``render`` writes tracked, text-only results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
import numpy as np

from moljax.conditioning.pseudospectra import arnoldi
from moljax.experimental.pme_conditioning import _counted_gmres

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path.home() / "moljax-artifacts/cn-benchmark-audit"
RESULTS = ROOT / "benchmarks/results"
CONFIGS: dict[str, dict[str, Any]] = {
    "schnakenberg": {
        "script": "benchmark_schnakenberg.py",
        "function": "cn_newton_step_snak",
        "n": 256,
        "dt": 0.01,
        "t_final": 100.0,
        "du": 1.0,
        "dv": 40.0,
        "a": 0.1,
        "b": 0.9,
        "gamma": 1000.0,
        "tol": 1e-8,
        "cap": 15,
    },
    "brusselator": {
        "script": "benchmark_brusselator.py",
        "function": "cn_newton_step_brus",
        "n": 256,
        "dt": 0.05,
        "t_final": 50.0,
        "du": 0.01,
        "dv": 0.02,
        "a": 1.0,
        "b": 3.4,
        "tol": 1e-8,
        "cap": 15,
    },
    "gray_scott": {
        "script": "benchmark_gray_scott.py",
        "function": "cn_newton_step_jit",
        "n": 256,
        "dt": 0.5,
        "t_final": 10000.0,
        "du": 2e-5,
        "dv": 1e-5,
        "feed": 0.035,
        "kill": 0.065,
        "tol": 1e-8,
        "cap": 10,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lap_symbol(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n, 1.0 / n) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return -(kx**2 + ky**2)


def initial_state(name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = CONFIGS[name]
    rng = np.random.RandomState(42)
    if name == "schnakenberg":
        ustar = cfg["a"] + cfg["b"]
        vstar = cfg["b"] / ustar**2
        return ustar + 0.01 * rng.randn(n, n), vstar + 0.01 * rng.randn(n, n)
    if name == "brusselator":
        return cfg["a"] + 0.01 * rng.randn(n, n), cfg["b"] / cfg["a"] + 0.01 * rng.randn(n, n)
    u = np.ones((n, n)) - 0.5 * rng.rand(n, n) * 0.1
    v = 0.25 * np.ones((n, n)) + 0.5 * rng.rand(n, n) * 0.1
    centre, radius = n // 2, 10
    u[centre - radius : centre + radius, centre - radius : centre + radius] = 0.5
    v[centre - radius : centre + radius, centre - radius : centre + radius] = 0.25
    return u, v


def homogeneous(name: str) -> tuple[float, float]:
    cfg = CONFIGS[name]
    if name == "schnakenberg":
        u = cfg["a"] + cfg["b"]
        return u, cfg["b"] / u**2
    if name == "brusselator":
        return cfg["a"], cfg["b"] / cfg["a"]
    return 1.0, 0.0


def reaction(name: str, u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    cfg = CONFIGS[name]
    if name == "schnakenberg":
        u2v = u * u * v
        return cfg["gamma"] * (cfg["a"] - u + u2v), cfg["gamma"] * (cfg["b"] - u2v)
    if name == "brusselator":
        u2v = u * u * v
        return cfg["a"] - (cfg["b"] + 1.0) * u + u2v, cfg["b"] * u - u2v
    uvv = u * v * v
    return -uvv + cfg["feed"] * (1 - u), uvv - (cfg["feed"] + cfg["kill"]) * v


def reaction_jacobian(name: str, u: jax.Array, v: jax.Array) -> tuple[jax.Array, ...]:
    cfg = CONFIGS[name]
    if name == "schnakenberg":
        g = cfg["gamma"]
        return g * (-1 + 2 * u * v), g * u * u, -2 * g * u * v, -g * u * u
    if name == "brusselator":
        return -(cfg["b"] + 1.0) + 2 * u * v, u * u, cfg["b"] - 2 * u * v, -(u * u)
    return -(v * v) - cfg["feed"], -2 * u * v, v * v, 2 * u * v - cfg["feed"] - cfg["kill"]


def _blocks(name: str, lam: jax.Array, mean: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    cfg = CONFIGS[name]
    h = 0.5 * cfg["dt"]
    a, b, c, d = mean
    return 1 - h * (cfg["du"] * lam + a), -h * b, -h * c, 1 - h * (cfg["dv"] * lam + d)


def _minimum_singular(blocks: tuple[jax.Array, ...]) -> jax.Array:
    a, b, c, d = blocks
    squared = a * a + b * b + c * c + d * d
    discriminant = jnp.sqrt((a * a + c * c - b * b - d * d) ** 2 + 4 * (a * b + c * d) ** 2)
    sigma_max = jnp.sqrt((squared + discriminant) / 2)
    return jnp.min(jnp.abs(a * d - b * c) / sigma_max)


def _inverse_apply(
    blocks: tuple[jax.Array, ...], x: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    a, b, c, d = blocks
    xhat, yhat = jnp.fft.fft2(x[0]), jnp.fft.fft2(x[1])
    determinant = a * d - b * c
    return (
        jnp.real(jnp.fft.ifft2((d * xhat - b * yhat) / determinant)),
        jnp.real(jnp.fft.ifft2((-c * xhat + a * yhat) / determinant)),
    )


def make_step(name: str, n: int, arm: str):
    """Return the script's while-loop update and extra, read-only diagnostics."""
    cfg = CONFIGS[name]
    lam = jnp.asarray(lap_symbol(n))
    h = 0.5 * cfg["dt"]
    p_u = 1.0 / (1.0 - h * cfg["du"] * lam)
    p_v = 1.0 / (1.0 - h * cfg["dv"] * lam)

    def residual(old, new):
        u_old, v_old = old
        u_new, v_new = new
        lap_u_new = jnp.real(jnp.fft.ifft2(lam * jnp.fft.fft2(u_new)))
        lap_v_new = jnp.real(jnp.fft.ifft2(lam * jnp.fft.fft2(v_new)))
        lap_u_old = jnp.real(jnp.fft.ifft2(lam * jnp.fft.fft2(u_old)))
        lap_v_old = jnp.real(jnp.fft.ifft2(lam * jnp.fft.fft2(v_old)))
        ru_new, rv_new = reaction(name, u_new, v_new)
        ru_old, rv_old = reaction(name, u_old, v_old)
        return (
            u_new - u_old - h * (cfg["du"] * (lap_u_new + lap_u_old) + ru_new + ru_old),
            v_new - v_old - h * (cfg["dv"] * (lap_v_new + lap_v_old) + rv_new + rv_old),
        )

    def step(old):
        if arm == "mean_field":
            mean = tuple(jnp.mean(x) for x in reaction_jacobian(name, *old))
            blocks = _blocks(name, lam, mean)
            min_sigma = _minimum_singular(blocks)
        else:
            blocks = None
            min_sigma = jnp.array(jnp.nan)

        def apply_precond(du, dv):
            if arm == "mean_field":
                return _inverse_apply(blocks, (du, dv))
            du_hat = jnp.fft.fft2(du)
            dv_hat = jnp.fft.fft2(dv)
            return jnp.real(jnp.fft.ifft2(p_u * du_hat)), jnp.real(jnp.fft.ifft2(p_v * dv_hat))

        def cond_fn(state):
            _, _, lagged, i = state
            return (lagged > cfg["tol"]) & (i < cfg["cap"])

        def body_fn(state):
            u_new, v_new, _, i = state
            res_u, res_v = residual(old, (u_new, v_new))
            du, dv = apply_precond(-res_u, -res_v)
            u_new = u_new + du
            v_new = v_new + dv
            lagged = jnp.max(jnp.abs(res_u)) + jnp.max(jnp.abs(res_v))
            return u_new, v_new, lagged, i + 1

        final = jax.lax.while_loop(cond_fn, body_fn, (old[0], old[1], jnp.array(1.0), 0))
        new = final[:2]
        post_u, post_v = residual(old, new)
        post = jnp.max(jnp.abs(post_u)) + jnp.max(jnp.abs(post_v))
        finite = jnp.all(jnp.isfinite(new[0])) & jnp.all(jnp.isfinite(new[1]))
        finite = finite & jnp.isfinite(final[2]) & jnp.isfinite(post)
        return new, (final[3], final[2], post, finite, min_sigma)

    return step, residual


def _float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _save_partial(artifacts: Path, payload: dict[str, Any]) -> None:
    path = artifacts / "measurements.json"
    temporary = artifacts / "measurements.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_partial(artifacts: Path) -> dict[str, Any]:
    path = artifacts / "measurements.json"
    return (
        json.loads(path.read_text()) if path.exists() else {"m0": {}, "m1": {}, "m2": {}, "m3": {}}
    )


def rho_2x2(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Spectral radius of real 2×2 blocks, including complex-conjugate roots."""
    trace = a + d
    root = np.lib.scimath.sqrt((a - d) ** 2 + 4 * b * c)
    return np.maximum(np.abs((trace + root) / 2), np.abs((trace - root) / 2))


def m0(name: str) -> dict[str, Any]:
    cfg = CONFIGS[name]
    u, v = homogeneous(name)
    j = np.array(
        [float(np.asarray(x)) for x in reaction_jacobian(name, jnp.array(u), jnp.array(v))]
    ).reshape(2, 2)
    lam = lap_symbol(cfg["n"])
    h = 0.5 * cfg["dt"]
    pu = 1.0 / (1.0 - h * cfg["du"] * lam)
    pv = 1.0 / (1.0 - h * cfg["dv"] * lam)
    a = h * pu * j[0, 0]
    b = h * pu * j[0, 1]
    c = h * pv * j[1, 0]
    d = h * pv * j[1, 1]
    rho = rho_2x2(a, b, c, d)
    mean_blocks = tuple(
        np.asarray(x)
        for x in _blocks(name, jnp.asarray(lam), tuple(jnp.asarray(x) for x in j.ravel()))
    )
    matrix = np.stack(
        (
            np.stack(
                (
                    np.broadcast_to(mean_blocks[0], lam.shape),
                    np.broadcast_to(mean_blocks[1], lam.shape),
                ),
                axis=-1,
            ),
            np.stack(
                (
                    np.broadcast_to(mean_blocks[2], lam.shape),
                    np.broadcast_to(mean_blocks[3], lam.shape),
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    mean_error = np.eye(2) - np.linalg.solve(matrix, matrix)
    return {
        "homogeneous_state": [u, v],
        "reaction_jacobian": j.tolist(),
        "rho_zero_mode": float(rho[0, 0]),
        "rho_max": float(np.max(rho)),
        "modes_rho_ge_1": int(np.count_nonzero(rho >= 1.0)),
        "mean_field_error_max_abs": float(np.max(np.abs(mean_error))),
        "mean_field_blocks_nonsingular": bool(
            float(_minimum_singular(tuple(jnp.asarray(x) for x in mean_blocks))) > 0
        ),
        "grid_modes": int(lam.size),
    }


def _classify(records: list[dict[str, Any]], tolerance: float, cap: int) -> str:
    if all(r["exit_reason"] == "tolerance" and r["post_residual"] <= tolerance for r in records):
        return "SOLVES"
    bad_caps = sum(
        r["exit_reason"] == "cap" and (r["post_residual"] or float("inf")) > 1e-6 for r in records
    )
    if any(not r["finite"] for r in records) or bad_caps / len(records) > 0.01:
        return "DOES NOT SOLVE"
    return "MARGINAL"


def run_arm(name: str, arm: str, artifacts: Path) -> dict[str, Any]:
    cfg = CONFIGS[name]
    n = cfg["n"]
    total = int(np.ceil(cfg["t_final"] / cfg["dt"]))
    state = tuple(jnp.asarray(x) for x in initial_state(name, n))
    step, _ = make_step(name, n, arm)
    chunk = jax.jit(
        lambda uv, count: jax.lax.scan(lambda x, _: step(x), uv, None, length=count),
        static_argnums=1,
    )
    records: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    elapsed_start = time.monotonic()
    current = 0
    while current < total:
        next_snapshot = min((s for s in (1, 10, 100, total) if s > current), default=total)
        length = min(200, total - current, next_snapshot - current)
        state, observations = chunk(state, length)
        iterations, lagged, post, finite, sigma = (
            np.asarray(x) for x in jax.device_get(observations)
        )
        for j in range(length):
            reason = (
                "non-finite"
                if not finite[j]
                else (
                    "cap" if iterations[j] >= cfg["cap"] and lagged[j] > cfg["tol"] else "tolerance"
                )
            )
            records.append(
                {
                    "step": current + j + 1,
                    "iterations": int(iterations[j]),
                    "exit_reason": reason,
                    "lagged_residual": _float(lagged[j]),
                    "post_residual": _float(post[j]),
                    "finite": bool(finite[j]),
                    "minimum_block_sigma": _float(sigma[j]) if arm == "mean_field" else None,
                }
            )
        current += length
        if current in (1, 10, 100, total):
            path = artifacts / f"{name}_{arm}_step_{current}.npz"
            np.savez(path, u=np.asarray(state[0]), v=np.asarray(state[1]))
            selected[str(current)] = {"path": str(path), "sha256": sha256(path)}
        if current % 1000 == 0 or current == total:
            print(
                f"{name} {arm}: {current}/{total} steps, {time.monotonic()-elapsed_start:.1f} CPU seconds",
                flush=True,
            )
    counts = [r["iterations"] for r in records]
    return {
        "category": _classify(records, cfg["tol"], cfg["cap"]),
        "steps": total,
        "first_nonfinite_step": next((r["step"] for r in records if not r["finite"]), None),
        "iteration_stats": {
            "min": min(counts),
            "median": float(np.median(counts)),
            "mean": float(np.mean(counts)),
            "max": max(counts),
        },
        "records": records,
        "selected_states": selected,
        "elapsed_cpu_seconds": time.monotonic() - elapsed_start,
    }


def _schnakenberg_etd_states(artifacts: Path) -> dict[str, dict[str, Any]]:
    """Use the paper's Cox–Matthews ETDRK4 coefficients and stages, at dt=5e-4."""
    cfg = CONFIGS["schnakenberg"]
    n = cfg["n"]
    dt = 5e-4
    lam = lap_symbol(n)
    circle = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)

    def coefficients(diffusivity):
        z = dt * diffusivity * lam[..., None] + circle
        return tuple(
            jnp.asarray(x)
            for x in (
                np.exp(dt * diffusivity * lam),
                np.exp(dt * diffusivity * lam / 2),
                dt * np.mean((np.exp(z / 2) - 1) / z, axis=-1),
                dt * np.mean((-4 - z + np.exp(z) * (4 - 3 * z + z**2)) / z**3, axis=-1),
                dt * np.mean((2 + z + np.exp(z) * (-2 + z)) / z**3, axis=-1),
                dt * np.mean((-4 - 3 * z - z**2 + np.exp(z) * (4 - z)) / z**3, axis=-1),
            )
        )

    eu, ehu, qu, f1u, f2u, f3u = coefficients(cfg["du"])
    ev, ehv, qv, f1v, f2v, f3v = coefficients(cfg["dv"])

    def step(state):
        u, v = state
        uh, vh = jnp.fft.fft2(u), jnp.fft.fft2(v)
        ru, rv = reaction("schnakenberg", u, v)
        nu, nv = jnp.fft.fft2(ru), jnp.fft.fft2(rv)
        au, av = ehu * uh + qu * nu, ehv * vh + qv * nv
        aru, arv = reaction(
            "schnakenberg", jnp.real(jnp.fft.ifft2(au)), jnp.real(jnp.fft.ifft2(av))
        )
        nau, nav = jnp.fft.fft2(aru), jnp.fft.fft2(arv)
        bu, bv = ehu * uh + qu * nau, ehv * vh + qv * nav
        bru, brv = reaction(
            "schnakenberg", jnp.real(jnp.fft.ifft2(bu)), jnp.real(jnp.fft.ifft2(bv))
        )
        nbu, nbv = jnp.fft.fft2(bru), jnp.fft.fft2(brv)
        cu, cv = ehu * au + qu * (2 * nbu - nu), ehv * av + qv * (2 * nbv - nv)
        cru, crv = reaction(
            "schnakenberg", jnp.real(jnp.fft.ifft2(cu)), jnp.real(jnp.fft.ifft2(cv))
        )
        ncu, ncv = jnp.fft.fft2(cru), jnp.fft.fft2(crv)
        return (
            jnp.real(jnp.fft.ifft2(eu * uh + f1u * nu + f2u * (nau + nbu) + f3u * ncu)),
            jnp.real(jnp.fft.ifft2(ev * vh + f1v * nv + f2v * (nav + nbv) + f3v * ncv)),
        )

    advance = jax.jit(
        lambda state, count: jax.lax.fori_loop(0, count, lambda _, uv: step(uv), state),
        static_argnums=1,
    )
    state = tuple(jnp.asarray(x) for x in initial_state("schnakenberg", n))
    selected: dict[str, dict[str, Any]] = {}
    current = 0
    for target in (0, 20, 200, 2000):
        if target > current:
            state = advance(state, target - current)
        current = target
        path = artifacts / f"schnakenberg_etdrk4_step_{target}.npz"
        np.savez(path, u=np.asarray(state[0]), v=np.asarray(state[1]))
        selected[str(target)] = {"path": str(path), "sha256": sha256(path), "t": target * dt}
    return selected


def _condition_number(blocks: tuple[np.ndarray, ...]) -> float:
    a, b, c, d = np.broadcast_arrays(*blocks)
    matrices = np.stack((np.stack((a, b), axis=-1), np.stack((c, d), axis=-1)), axis=-2)
    singular = np.linalg.svd(matrices, compute_uv=False)
    return float(np.max(singular[..., 0]) / np.min(singular[..., 1]))


def run_m3(index: int, artifacts: Path, partial: dict[str, Any]) -> dict[str, Any]:
    cfg = CONFIGS["schnakenberg"]
    arm = next(
        (a for a in ("m1", "m2") if partial[a]["schnakenberg"]["category"] == "SOLVES"), None
    )
    if arm is None:
        if "etd_states" not in partial:
            partial["etd_states"] = _schnakenberg_etd_states(artifacts)
            _save_partial(artifacts, partial)
        selected = partial["etd_states"]
        step_number = (0, 20, 200, 2000)[index]
        source = "transient-only ETDRK4"
    else:
        selected = partial[arm]["schnakenberg"]["selected_states"]
        step_number = (1, 10, 100, partial[arm]["schnakenberg"]["steps"])[index]
        source = f"SOLVES {arm}"
    metadata = selected[str(step_number)]
    path = Path(metadata["path"])
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"state SHA-256 mismatch: {path}")
    with np.load(path) as data:
        state = (jnp.asarray(data["u"]), jnp.asarray(data["v"]))
    n = cfg["n"]
    lam = jnp.asarray(lap_symbol(n))
    h = 0.5 * cfg["dt"]
    reaction_entries = reaction_jacobian("schnakenberg", *state)
    mean = tuple(jnp.mean(x) for x in reaction_entries)
    mf_blocks = _blocks("schnakenberg", lam, mean)
    fft_blocks = (1 - h * cfg["du"] * lam, jnp.array(0.0), jnp.array(0.0), 1 - h * cfg["dv"] * lam)
    _, residual = make_step("schnakenberg", n, "fft")
    rhs = -jnp.concatenate([x.ravel() for x in residual(state, state)])

    @jax.jit
    def j_action(vector):
        du, dv = jnp.reshape(vector, (2, n, n))
        lu = jnp.real(jnp.fft.ifft2(lam * jnp.fft.fft2(du)))
        lv = jnp.real(jnp.fft.ifft2(lam * jnp.fft.fft2(dv)))
        a, b, c, d = reaction_entries
        return jnp.concatenate(
            (
                (du - h * (cfg["du"] * lu + a * du + b * dv)).ravel(),
                (dv - h * (cfg["dv"] * lv + c * du + d * dv)).ravel(),
            )
        )

    identity = _counted_gmres(j_action, rhs, tol=1e-8, max_iters=100)
    out: dict[str, Any] = {
        "state_source": source,
        "state_step": step_number,
        "state": metadata,
        "identity": identity,
        "preconditioners": {},
    }
    for label, blocks in (("fft", fft_blocks), ("mean_field", mf_blocks)):
        inverse = jax.jit(
            lambda x: jnp.concatenate(
                [y.ravel() for y in _inverse_apply(blocks, tuple(jnp.reshape(x, (2, n, n))))]
            )
        )
        left_action = jax.jit(lambda x: inverse(j_action(x)))
        right_action = jax.jit(lambda x: j_action(inverse(x)))
        left = _counted_gmres(left_action, inverse(rhs), tol=1e-8, max_iters=100)
        right = _counted_gmres(right_action, rhs, tol=1e-8, max_iters=100)
        projected = arnoldi(jax.jit(lambda x: x - left_action(x)), rhs, 50)
        eigenvalues = np.linalg.eigvals(
            np.asarray(projected.hessenberg)[: projected.k_achieved, : projected.k_achieved]
        )
        out["preconditioners"][label] = {
            "left": left,
            "right": right,
            "kappa_2_P": _condition_number(tuple(np.asarray(x) for x in blocks)),
            "arnoldi_rho_estimate": float(np.max(np.abs(eigenvalues))),
            "arnoldi_k_achieved": projected.k_achieved,
            "arnoldi_source": "moljax.conditioning.pseudospectra.arnoldi",
        }
    return out


def _pointer(value: Any, path: str) -> str:
    return f"{value} (`{path}`)"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# CN benchmark convergence audit",
        "",
        "CPU-only convergence evidence. Times in the JSON are execution logs, not benchmark timings.",
        "Every value below cites its JSON pointer in `cn_benchmark_audit.json`.",
        "",
        "## M0: homogeneous error matrix",
        "",
        "| Model | rho(G0) | max rho(Gk) | modes rho >= 1 | mean-field error max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in payload["m0"].items():
        p = f"/m0/{name}"
        lines.append(
            f"| {name} | {_pointer(item['rho_zero_mode'], p+'/rho_zero_mode')} | "
            f"{_pointer(item['rho_max'], p+'/rho_max')} | "
            f"{_pointer(item['modes_rho_ge_1'], p+'/modes_rho_ge_1')} | "
            f"{_pointer(item['mean_field_error_max_abs'], p+'/mean_field_error_max_abs')} |"
        )
    lines += [
        "",
        "## M1 and M2: CN trajectories",
        "",
        "| Arm | Model | Category | Steps | First non-finite | Iterations min/median/mean/max |",
        "|---|---|---|---:|---:|---|",
    ]
    for arm in ("m1", "m2"):
        for name, item in payload[arm].items():
            p = f"/{arm}/{name}"
            stats = item["iteration_stats"]
            joined = ", ".join(
                _pointer(stats[key], p + "/iteration_stats/" + key)
                for key in ("min", "median", "mean", "max")
            )
            lines.append(
                f"| {arm} | {name} | {_pointer(item['category'], p+'/category')} | {_pointer(item['steps'], p+'/steps')} | {_pointer(item['first_nonfinite_step'], p+'/first_nonfinite_step')} | {joined} |"
            )
    lines += [
        "",
        "Per-step exit reason, lagged and post-update residual, and finiteness are in `/m1/*/records/*` and `/m2/*/records/*`.",
        "The paper's Table 14 footnote uses approximately three NFE per CN step; the measured iteration distributions above test that assumption.",
        "",
        "## M3: Schnakenberg first-iterate systems",
        "",
        "| State | Source | Preconditioner | Left iterations | Right iterations | kappa2(P) | Arnoldi rho |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for index, item in payload["m3"].items():
        p = f"/m3/{index}"
        for label, preconditioner in item["preconditioners"].items():
            q = p + "/preconditioners/" + label
            lines.append(
                f"| {_pointer(item['state_step'], p+'/state_step')} | {_pointer(item['state_source'], p+'/state_source')} | {label} | {_pointer(preconditioner['left']['iterations'], q+'/left/iterations')} | {_pointer(preconditioner['right']['iterations'], q+'/right/iterations')} | {_pointer(preconditioner['kappa_2_P'], q+'/kappa_2_P')} | {_pointer(preconditioner['arnoldi_rho_estimate'], q+'/arnoldi_rho_estimate')} |"
            )
        lines.append(
            f"| {_pointer(item['state_step'], p+'/state_step')} | identity | identity | {_pointer(item['identity']['iterations'], p+'/identity/iterations')} | same system | 1 | not evaluated |"
        )
    lines += [
        "",
        "State paths and SHA-256 hashes are recorded at `/m3/*/state`; binary state files remain outside Git.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("m0", "m1", "m2", "m3", "render"), required=True)
    parser.add_argument("--case", choices=tuple(CONFIGS))
    parser.add_argument("--state-index", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    artifacts = args.artifact_dir.expanduser().resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    partial = _load_partial(artifacts)
    if args.stage in ("m0", "m1", "m2"):
        if args.case is None:
            parser.error("--case is required for M0, M1 and M2")
        partial[args.stage][args.case] = (
            m0(args.case)
            if args.stage == "m0"
            else run_arm(args.case, "fft" if args.stage == "m1" else "mean_field", artifacts)
        )
        _save_partial(artifacts, partial)
    elif args.stage == "m3":
        if args.state_index is None:
            parser.error("--state-index is required for M3")
        if "schnakenberg" not in partial["m1"] or "schnakenberg" not in partial["m2"]:
            parser.error("M1 and M2 Schnakenberg runs must precede M3")
        partial["m3"][str(args.state_index)] = run_m3(args.state_index, artifacts, partial)
        _save_partial(artifacts, partial)
    else:
        for stage in ("m0", "m1", "m2"):
            if set(partial[stage]) != set(CONFIGS):
                raise ValueError(f"incomplete {stage} data")
        if set(partial["m3"]) != {"0", "1", "2", "3"}:
            raise ValueError("incomplete M3 data")
        partial["provenance"] = {
            "base": "d8d4432a4b03ede71db24e2beeb3d6812d3f9c9b",
            "host": platform.node(),
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "platform": jax.default_backend(),
            "script_sha256": sha256(Path(__file__)),
            "benchmark_sha256": {
                name: sha256(ROOT / "benchmarks" / cfg["script"]) for name, cfg in CONFIGS.items()
            },
        }
        json_path = RESULTS / "cn_benchmark_audit.json"
        md_path = RESULTS / "CN_BENCHMARK_AUDIT_RESULTS.md"
        json_path.write_text(json.dumps(partial, indent=2, sort_keys=True, allow_nan=False) + "\n")
        md_path.write_text(render(partial))
        print(json_path)
        print(md_path)


if __name__ == "__main__":
    main()
