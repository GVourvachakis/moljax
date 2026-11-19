"""Regenerate the six D0b endpoint references used by the D3 benchmark.

Run ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=. python benchmarks/d3_references.py``.
Each accepted step of an ETDRK4 adaptive run with ``rtol = atol = 1e-8`` is
split into 16 equal traced ETDRK4 substeps. The endpoint is host specific.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from benchmarks.d3_work_precision import CASES, setup
from moljax.core import fft_integrators as fi
from moljax.core.dt_policy import PIDParams
from moljax.core.utils import StatusCode

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/d3_references.npz"
MANIFEST = ROOT / "benchmarks/results/d3_references.json"
D0B_EST_ERR = {
    "P2_128": 6.68671919946478e-12,
    "P2_256": 6.098554387694976e-12,
    "P3_64": 1.062314379302776e-06,
    "P4_64": 8.218991363928938e-08,
    "P5_64": 1.4326020483655173e-07,
    "P5_128": 2.986317503545861e-07,
}
# Relative confirmation levels from the D3 v3.2 reference audit, retained with
# their provenance because the independent cross-check endpoints are not inputs.
D3_CONFIRMATION = {
    "P1_64": (0.0, "exact"),
    "P2_128": (1.586128187889244e-09, "self-convergence in D3 metric"),
    "P2_256": (1.7194977621261063e-09, "self-convergence in D3 metric"),
    "P3_64": (5.523578644838155e-07, "endpoint distance plus bounded self-error"),
    "P4_64": (2.8765795458517965e-08, "endpoint distance plus bounded self-error"),
    "P5_64": (9.065371217057927e-06, "endpoint distance plus bounded self-error"),
    "P5_128": (5.549261130598974e-07, "endpoint distance plus bounded self-error"),
}


def _accepted_steps(name: str, n: int):
    """Return the D0b ETDRK4 trajectory's accepted step sizes."""
    state, ops, rhs, _, final = setup(name, n)
    max_steps = 1024
    while True:
        result = fi.etd_integrate_adaptive(
            state,
            (0.0, final),
            1e-4 * final,
            ops,
            rhs,
            method="etdrk4",
            pid_params=PIDParams(atol=1e-8, rtol=1e-8, dt_max=final),
            max_steps=max_steps,
            save_every=1,
        )
        jax.tree.map(jax.block_until_ready, result.y_final)
        if int(result.status) != StatusCode.MAX_STEPS_REACHED:
            if not result.success:
                raise RuntimeError(f"reference source did not converge: {name}/{n}")
            break
        limit = int(1.5e9 // (len(state) * n * n * 8)) - 2
        next_steps = min(max_steps * 2, limit)
        if next_steps <= max_steps:
            raise MemoryError(f"reference history exceeds guard: {name}/{n}")
        max_steps = next_steps

    h = np.asarray(result.dt_history[1 : int(result.n_saved)], dtype=np.float64)
    if len(h) != int(result.n_accepted):
        raise RuntimeError(f"reference step history length differs: {name}/{n}")
    return state, ops, rhs, h


def _refined_endpoint(source, m: int) -> dict[str, np.ndarray]:
    state, ops, rhs, h = source
    substeps = jnp.asarray(np.repeat(h / m, m))

    @jax.jit
    def integrate(step_array):
        def body(carry, dt):
            y, t = carry
            y = fi._etdrk4_step_at(y, t, t + dt / 2, t + dt, dt, ops, rhs)
            return (y, t + dt), None

        return jax.lax.scan(body, (state, jnp.asarray(0.0)), step_array)[0]

    endpoint, _ = integrate(substeps)
    return {key: np.asarray(value) for key, value in endpoint.items()}


def estimate_error(name: str, n: int) -> tuple[float, dict[str, np.ndarray]]:
    """D0b B1: three refinements, observed order, then d2/(2**p - 1)."""
    source = _accepted_steps(name, n)
    endpoints = {m: _refined_endpoint(source, m) for m in (4, 8, 16)}

    def distance(a, b):
        left = np.concatenate([endpoints[a][key].ravel() for key in sorted(endpoints[a])])
        right = np.concatenate([endpoints[b][key].ravel() for key in sorted(endpoints[b])])
        return float(np.max(np.abs(left - right)))

    d1, d2 = distance(4, 8), distance(8, 16)
    p_obs = math.log2(d1 / d2) if d1 > 0 and d2 > 0 else None
    if p_obs is not None and 2 <= p_obs <= 6:
        est_err = d2 / (2**p_obs - 1)
    else:
        endpoints[32] = _refined_endpoint(source, 32)
        d3 = distance(16, 32)
        p32 = math.log2(d2 / d3) if d2 > 0 and d3 > 0 else None
        if p32 is None or not 2 <= p32 <= 6:
            raise RuntimeError(f"D0b B1 order outside [2, 6] after m=32: {name}/{n}: {p32}")
        est_err = d3 / (2**p32 - 1)
    case = f"{name}_{n}"
    if est_err != D0B_EST_ERR[case]:
        raise RuntimeError(f"D0b B1 estimate differs: {case}: {est_err} != {D0B_EST_ERR[case]}")
    return est_err, endpoints[16]


def generate_case(name: str, n: int) -> dict[str, np.ndarray]:
    """Reproduce D0b's accepted-step sequence and sixteen-way refinement."""
    return _refined_endpoint(_accepted_steps(name, n), 16)


def main() -> None:
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise RuntimeError("reference generation requires CPU and x64")
    arrays = {}
    manifest = {
        "recipe": "D0b ETDRK4 rtol=atol=1e-8 accepted steps; each divided into 16 traced ETDRK4 substeps",
        "host_specific": True,
        "est_err_recipe": "D0b B1: for each accepted ETDRK4 step at rtol=atol=1e-8, split it into 4, 8 and 16 traced substeps; d1=||ref4-ref8||_inf, d2=||ref8-ref16||_inf, p=log2(d1/d2), est_err=d2/(2**p-1) when 2<=p<=6; otherwise repeat with 32 and use d3=||ref16-ref32||_inf, p32=log2(d2/d3); matches D0b B1 bitwise",
        "confirmation_source": "D3 v3.2 reference audit on experiment/etd-embedded-pairs",
        "confirmation": {
            case: {"epsilon_conf": value, "method": method}
            for case, (value, method) in D3_CONFIRMATION.items()
        },
        "cases": {},
    }
    for name, n in CASES:
        if name == "P1":
            continue
        est_err, endpoint = estimate_error(name, n)
        case = f"{name}_{n}"
        manifest["cases"][case] = {"est_err": est_err, "arrays": {}}
        for key, value in endpoint.items():
            arrays[f"{case}_{key}"] = value
            manifest["cases"][case]["arrays"][key] = hashlib.sha256(value.tobytes()).hexdigest()
        print(case, manifest["cases"][case], flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if OUTPUT.stat().st_size > 5_000_000:
        raise RuntimeError("reference archive exceeds 5 MB")
    print(f"archive bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
