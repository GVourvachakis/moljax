"""Regenerate the six D0b endpoint references used by the D3 benchmark.

Run ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=. .venv/bin/python benchmarks/d3_references.py``.
Each accepted step of an ETDRK4 adaptive run with ``rtol = atol = 1e-8`` is
split into 16 equal traced ETDRK4 substeps. The endpoint is host specific.
"""

from __future__ import annotations

import hashlib
import json
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
D0B_JSON_SHA256 = "a31665b32fdd5d31ea44c30492f70e0d82a51a2b4efcc0a9fa1bf8ecdc9d9faf"
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


def generate_case(name: str, n: int) -> dict[str, np.ndarray]:
    """Reproduce D0b's accepted-step sequence and its sixteen-way refinement."""
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
    substeps = jnp.asarray(np.repeat(h / 16, 16))

    @jax.jit
    def integrate(step_array):
        def body(carry, dt):
            y, t = carry
            y = fi._etdrk4_step_at(y, t, t + dt / 2, t + dt, dt, ops, rhs)
            return (y, t + dt), None

        return jax.lax.scan(body, (state, jnp.asarray(0.0)), step_array)[0]

    endpoint, _ = integrate(substeps)
    return {key: np.asarray(value) for key, value in endpoint.items()}


def main() -> None:
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise RuntimeError("reference generation requires CPU and x64")
    arrays = {}
    manifest = {
        "recipe": "D0b ETDRK4 rtol=atol=1e-8 accepted steps; each divided into 16 traced ETDRK4 substeps",
        "host_specific": True,
        "est_err_source_sha256": D0B_JSON_SHA256,
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
        endpoint = generate_case(name, n)
        case = f"{name}_{n}"
        manifest["cases"][case] = {"est_err": D0B_EST_ERR[case], "arrays": {}}
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
