"""Reproduce the frozen D3 adaptive-ETD work-precision study on CPU.

Run ``JAX_PLATFORMS=cpu PYTHONPATH="$PWD" python benchmarks/d3_work_precision.py all``.
Raw checkpoints and states stay in ~/moljax_scratch/d3; only the summary is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy

import moljax
from moljax.core.dt_policy import PIDParams
from moljax.core.etd_pairs import ERK32ZB, ERK43ZB, pair_step_at
from moljax.core.fft_integrators import _etdrk4_step_at, etd_integrate, etd_integrate_adaptive
from moljax.core.fft_operators import DiffusionOperator
from moljax.core.grid import Grid2D
from moljax.core.model import (
    create_brusselator_periodic_fft,
    create_gray_scott_periodic_fft,
    create_schnakenberg_periodic_fft,
)
from moljax.core.state import scaled_error_norm_interior
from moljax.core.stepping import adaptive_integrate_imex
from moljax.core.utils import StatusCode
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    _initial_state,
    build_brusselator_system,
)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path.home() / "moljax_scratch/d3"
EVIDENCE = Path.home() / "moljax_scratch/d0b"
RAW = SCRATCH / "results/stage_a.jsonl"
CASES = (("P1", 64), ("P2", 128), ("P2", 256), ("P3", 64), ("P4", 64), ("P5", 64), ("P5", 128))
RTOLS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8)
CONFIGS = ("S0", "S1a", "S1b", "S1c", "E0a", "E0b", "X", "sensitivity")
NFE = {"S0": 12, "S1a": 12, "S1b": 12, "S1c": 12, "sensitivity": 12, "E0a": 4, "E0b": 5}
TRANSFORMS_PER_FIELD = {
    "S0": 27,
    "S1a": 27,
    "S1b": 27,
    "S1c": 27,
    "sensitivity": 27,
    "E0a": 9,
    "E0b": 11,
}
CONTROLLERS = {
    "S0": None,
    "S1a": "PI34",
    "S1b": "PI42",
    "S1c": "H312PID",
    "sensitivity": None,
    "E0a": None,
    "E0b": None,
}
METHODS = {"E0a": "erk32zb", "E0b": "erk43zb"}
MEMORY_LIMIT = 1.5e9


def check_environment():
    expected = ROOT / "moljax/__init__.py"
    if Path(moljax.__file__).resolve() != expected.resolve():
        raise RuntimeError(f"moljax imported from {moljax.__file__}, expected {expected}")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise RuntimeError("D3 requires CPU and x64")
    if (jax.__version__, jaxlib.__version__) != ("0.11.2", "0.11.2"):
        raise RuntimeError("D3 requires the D0b-matching JAX/jaxlib 0.11.2 environment")
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "backend": jax.default_backend(),
        "moljax_file": str(Path(moljax.__file__).resolve()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "x64": bool(jax.config.jax_enable_x64),
        "execution_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "AC_online": (
            Path("/sys/class/power_supply/ACAD/online").read_text().strip()
            if Path("/sys/class/power_supply/ACAD/online").exists()
            else None
        ),
    }


def verify_references():
    manifest = EVIDENCE / "results/SHA256SUMS"
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = EVIDENCE / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"D0b reference manifest mismatch: {path}")


def setup(name, n):
    """D0 problem definitions: FD DiffusionOperator and the paper IC recipes."""
    if name == "P1":
        length, final, ds = 1.0, 2.0, {"u": 0.01}
    elif name == "P2":
        length, final, ds = 1.0, 100.0, {"u": 2e-5, "v": 1e-5}
    elif name == "P3":
        length, final, ds = 5.0, 200.0, {"u": 0.01, "v": 0.1}
    elif name == "P4":
        length, final, ds = 5.0, 20.0, {"u": 0.01, "v": 0.02}
    elif name == "P5":
        length, final, ds = 1.0, 1.0, {"u": 1.0, "v": 40.0}
    else:
        raise ValueError(name)
    grid = Grid2D.uniform(n, n, 0.0, length, 0.0, length)
    ops = {key: DiffusionOperator(grid, d, dtype=jnp.float64) for key, d in ds.items()}
    if name == "P1":
        coords = np.arange(n) * length / n
        s = jnp.asarray(np.outer(np.sin(2 * np.pi * coords), np.sin(2 * np.pi * coords)))
        ls = ops["u"].matvec(s)
        eigenvalue = float(jnp.vdot(s, ls) / jnp.vdot(s, s))
        assert float(jnp.max(jnp.abs(ls - eigenvalue * s))) / float(jnp.max(jnp.abs(ls))) < 1e-12

        def exact(t):
            return {"u": (1 + 0.5 * jnp.cos(t)) * s}

        def rhs(state, t):
            target = exact(t)["u"]
            forcing = -0.5 * jnp.sin(t) * s - eigenvalue * target + target**3
            return {"u": -(state["u"] ** 3) + forcing}

        y0 = exact(0.0)
    elif name == "P2":
        np.random.seed(42)
        u = np.ones((n, n)) - 0.5 * np.random.rand(n, n) * 0.1
        v = 0.25 * np.ones((n, n)) + 0.5 * np.random.rand(n, n) * 0.1
        cx, cy = n // 2, n // 2
        u[cx - 10 : cx + 10, cy - 10 : cy + 10] = 0.5
        v[cx - 10 : cx + 10, cy - 10 : cy + 10] = 0.25
        y0 = {"u": jnp.array(u), "v": jnp.array(v)}

        def rhs(y, t):
            uvv = y["u"] * y["v"] ** 2
            return {"u": -uvv + 0.035 * (1 - y["u"]), "v": uvv - 0.1 * y["v"]}

        exact = None
    elif name in ("P3", "P4"):
        regime = TURING_REGIME if name == "P3" else HOPF_REGIME
        model, _, _ = build_brusselator_system(regime, grid)
        seed = 20260822 + (name == "P3")
        full = _initial_state(model, regime, 1e-3, seed)
        iy, ix = grid.interior_slice
        y0 = {key: value[iy, ix] for key, value in full.items()}

        def rhs(y, t):
            uv = y["u"] ** 2 * y["v"]
            return {"u": regime.a - (regime.b + 1) * y["u"] + uv, "v": regime.b * y["u"] - uv}

        exact = None
    else:
        np.random.seed(42)
        u = 1.0 + 0.01 * np.random.randn(n, n)
        v = 0.9 + 0.01 * np.random.randn(n, n)
        y0 = {"u": jnp.array(u), "v": jnp.array(v)}

        def rhs(y, t):
            uv = y["u"] ** 2 * y["v"]
            return {"u": 1000 * (0.1 - y["u"] + uv), "v": 1000 * (0.9 - uv)}

        exact = None
    return y0, ops, rhs, exact, final


def reference(name, n, exact, final):
    if exact is not None:
        return {key: np.asarray(value) for key, value in exact(final).items()}
    path = EVIDENCE / f"refs/{name}_{n}.npz"
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def endpoint_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        array = np.asarray(state[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def error_summary(state, ref, name, n):
    norms = {key: float(np.max(np.abs(value))) for key, value in ref.items()}
    errors = {key: float(np.max(np.abs(np.asarray(state[key]) - ref[key]))) for key in ref}
    absolute = max(errors.values())
    relative = {key: errors[key] / norms[key] for key in ref}
    if name == "P1":
        validity = "valid (exact)"
    else:
        data = json.loads((EVIDENCE / "results/d0b.json").read_text())
        est = next(row["est_err"] for row in data["B1"] if row["problem"] == name and row["N"] == n)
        validity = (
            "valid"
            if absolute >= 100 * est
            else "valid, reference-limited"
            if absolute >= 10 * est
            else "indeterminate"
        )
    return {
        "error_inf_absolute": absolute,
        "error_inf_per_field": errors,
        "error_inf_relative_per_field": relative,
        "error_inf_relative": max(relative.values()),
        "validity": validity,
    }


def history_settings(n, n_fields, max_steps):
    save_every = max(
        1, math.ceil(max_steps / max(1, int(MEMORY_LIMIT // (n_fields * n * n * 8)) - 2))
    )
    hist_bytes = (max_steps // save_every + 2) * n_fields * n * n * 8
    if hist_bytes > MEMORY_LIMIT:
        raise MemoryError(f"history {hist_bytes} > {MEMORY_LIMIT}")
    return save_every, hist_bytes


def roughness(result, save_every):
    h = np.asarray(result.dt_history[1 : int(result.n_saved)])
    if len(h) < 3 or not np.all(h > 0):
        return {"status": "unavailable", "save_every": save_every}
    ramp_end = int(np.argmax(h >= 0.5 * np.median(h)))
    ratios = h[ramp_end + 1 :] / h[ramp_end:-1]
    if len(ratios) == 0:
        return {"status": "unavailable after ramp", "save_every": save_every}
    log_ratios = np.abs(np.log2(ratios))
    return {
        "status": "exact" if save_every == 1 else "sampled saved-step ratios",
        "save_every": save_every,
        "ramp_end_saved_index": ramp_end,
        "median_abs_log2_ratio": float(np.median(log_ratios)),
        "max_abs_log2_ratio": float(np.max(log_ratios)),
    }


def imex_system(name, n, y0):
    length = 5.0 if name in ("P3", "P4") else 1.0
    grid = Grid2D.uniform(n, n, 0.0, length, 0.0, length, n_ghost=1)
    if name == "P2":
        model, cache, diffusivities = create_gray_scott_periodic_fft(
            grid, Du=2e-5, Dv=1e-5, F=0.035, k=0.065
        )
    elif name in ("P3", "P4"):
        model, cache, diffusivities = create_brusselator_periodic_fft(
            grid, Du=0.01, Dv=0.1 if name == "P3" else 0.02, a=1.0, b=1.8 if name == "P3" else 3.4
        )
    elif name == "P5":
        model, cache, diffusivities = create_schnakenberg_periodic_fft(grid)
    else:
        raise ValueError("X is not defined for P1")
    iy, ix = grid.interior_slice
    padded = {
        key: jnp.zeros((grid.ny_total, grid.nx_total), dtype=jnp.float64).at[iy, ix].set(value)
        for key, value in y0.items()
    }
    return model, cache, diffusivities, model.apply_bcs(padded, 0.0)


def adaptive_cell(name, n, config, rtol, rhs_uses_time=None):
    y0, ops, rhs, exact, final = setup(name, n)
    pid = PIDParams(atol=rtol if config == "sensitivity" else 1e-3 * rtol, rtol=rtol, dt_max=final)
    if config == "X":
        model, cache, diffusivities, padded = imex_system(name, n, y0)
        max_steps = save_every = 100_000
        hist_bytes = (max_steps // save_every + 2) * len(y0) * n * n * 8
        if hist_bytes > MEMORY_LIMIT:
            raise MemoryError("X history exceeds the memory guard")

        def solve():
            return adaptive_integrate_imex(
                model,
                padded,
                0.0,
                final,
                1e-4 * final,
                cache,
                diffusivities,
                use_strang=True,
                pid_params=pid,
                max_steps=max_steps,
                save_every=save_every,
            )
    else:
        selection = (
            json.loads((SCRATCH / "results/e1_selection.json").read_text())
            if config == "E1"
            else None
        )
        method = selection["method"] if selection else METHODS.get(config, "etdrk4")
        controller = selection["controller"] if selection else CONTROLLERS[config]
        uses_time = method != "erk43zb" if rhs_uses_time is None else rhs_uses_time
        max_steps = 1024

        def solve(max_steps, save_every):
            return etd_integrate_adaptive(
                y0,
                (0.0, final),
                1e-4 * final,
                ops,
                rhs,
                method=method,
                rhs_uses_time=uses_time,
                pid_params=pid,
                controller=controller,
                max_steps=max_steps,
                save_every=save_every,
            )

    if config == "X":
        first = solve()
        jax.tree.map(jax.block_until_ready, first.y_final)
        escalations = []
    else:
        escalations = []
        while True:
            save_every, hist_bytes = history_settings(n, len(y0), max_steps)
            first = solve(max_steps, save_every)
            jax.tree.map(jax.block_until_ready, first.y_final)
            if int(first.status) != StatusCode.MAX_STEPS_REACHED or max_steps == 200_000:
                break
            escalations.append(max_steps)
            max_steps = min(200_000, max_steps * 2)
    start = time.perf_counter()
    result = solve() if config == "X" else solve(max_steps, save_every)
    jax.tree.map(jax.block_until_ready, result.y_final)
    steady = time.perf_counter() - start
    if int(first.n_accepted) != int(result.n_accepted) or int(first.n_rejected) != int(
        result.n_rejected
    ):
        raise RuntimeError("warmup and steady counts differ")
    accepted, rejected = int(result.n_accepted), int(result.n_rejected)
    attempts = accepted + rejected
    state = (
        {key: value[model.grid.interior_slice] for key, value in result.y_final.items()}
        if config == "X"
        else result.y_final
    )
    success = bool(result.status == StatusCode.SUCCESS) if config == "X" else bool(result.success)
    row = {
        "problem": name,
        "N": n,
        "config": config,
        "rtol": rtol,
        "atol": pid.atol,
        "accepted": accepted,
        "rejected": rejected,
        "attempts": attempts,
        "success": success,
        "status": int(result.status),
        "endpoint_sha256": endpoint_sha(state),
        "steady_seconds": steady,
        "max_steps": max_steps,
        "save_every": save_every,
        "hist_bytes": hist_bytes,
        "capacity_escalations": escalations,
        "rhs_uses_time": uses_time if config != "X" else None,
        "provenance": check_environment(),
    }
    if config != "X":
        row["NFE"] = attempts * (NFE[config] if config != "E1" else 4 if method == "erk32zb" else 5)
        row["transforms"] = (
            attempts
            * (TRANSFORMS_PER_FIELD[config] if config != "E1" else 9 if method == "erk32zb" else 11)
            * len(y0)
        )
        row["lean_NFE"] = (
            (10 * attempts + accepted)
            if method == "etdrk4"
            else (3 * attempts + 1)
            if method == "erk32zb"
            else (4 * attempts + accepted)
        )
        row["roughness"] = roughness(result, save_every)
        if selection:
            row["selected_method"] = method
            row["selected_controller"] = controller
    else:
        row["NFE"] = 6 * attempts
        row["NFE_basis"] = (
            "six model.nonlinear_rhs calls per attempted IMEX-Strang step, D3.0c count"
        )
        row["lean_NFE"] = None
        row["transforms"] = None
        row["roughness"] = {"status": "not saved by the X history protocol"}
    if success:
        row.update(error_summary(state, reference(name, n, exact, final), name, n))
    else:
        row["reason"] = "integration did not reach the final time"
    return row


def saved_s0(name, n, rtol):
    y0, ops, rhs, _, final = setup(name, n)
    pid = PIDParams(rtol=rtol, atol=1e-3 * rtol, dt_max=final)
    max_steps = 1024
    while True:
        _, hist_bytes = history_settings(n, len(y0), max_steps)
        if hist_bytes > MEMORY_LIMIT:
            raise MemoryError("probe full history exceeds the memory guard")
        result = etd_integrate_adaptive(
            y0,
            (0.0, final),
            1e-4 * final,
            ops,
            rhs,
            method="etdrk4",
            pid_params=pid,
            max_steps=max_steps,
            save_every=1,
        )
        jax.tree.map(jax.block_until_ready, result.y_final)
        if int(result.status) != StatusCode.MAX_STEPS_REACHED or max_steps == 200_000:
            if not bool(result.success):
                raise RuntimeError(f"S0 probe source failed: {name}/{n} rtol={rtol}")
            return result, ops, rhs, final
        max_steps = min(max_steps * 2, 200_000)


def estimator_power_cell(name, n):
    source, ops, rhs, final = saved_s0(name, n, 1e-6)
    control, _, _, _ = saved_s0(name, n, 1e-4)
    h_med = float(np.median(np.asarray(control.dt_history[1 : int(control.n_saved)])))
    times = np.asarray(source.t_history[: int(source.n_saved)])
    methods = ("S0", "E0a") if name == "P1" else ("S0", "E0a", "E0b")
    entries = []
    for fraction in (0.05, 0.5, 0.95):
        index = int(np.argmin(np.abs(times - fraction * final)))
        t = float(times[index])
        y = {key: value[index] for key, value in source.y_history.items()}
        for method in methods:
            norms = []
            for k in range(7):
                h = h_med * 2 ** (-k)
                if method == "S0":
                    full = _etdrk4_step_at(y, t, t + h / 2, t + h, h, ops, rhs)
                    half1 = _etdrk4_step_at(y, t, t + h / 4, t + h / 2, h / 2, ops, rhs)
                    high = _etdrk4_step_at(half1, t + h / 2, t + 3 * h / 4, t + h, h / 2, ops, rhs)
                    err = jax.tree.map(lambda a, b: (a - b) / 15, high, full)
                else:
                    tableau = ERK32ZB if method == "E0a" else ERK43ZB
                    stage_times = tuple(t + float(c) * h for c in tableau.c)
                    high, low = pair_step_at(y, stage_times, h, ops, rhs, tableau)
                    err = jax.tree.map(lambda a, b: a - b, high, low)
                norms.append(float(scaled_error_norm_interior(err, high, 1e-7, 1e-4)))
            slopes = [
                math.log2(norms[k] / norms[k + 1]) if norms[k + 1] > 0 else None for k in range(6)
            ]
            entries.append(
                {
                    "method": method,
                    "nominal_power": {"S0": 5, "E0a": 3, "E0b": 4}[method],
                    "target_fraction": fraction,
                    "saved_time": t,
                    "source_index": index,
                    "h_median": h_med,
                    "norms": norms,
                    "slopes": slopes,
                    "median_slope_k_ge_2": float(
                        np.median([v for v in slopes[2:] if v is not None])
                    ),
                }
            )
    return {
        "problem": name,
        "N": n,
        "source_counts": [int(source.n_accepted), int(source.n_rejected)],
        "control_counts": [int(control.n_accepted), int(control.n_rejected)],
        "entries": entries,
        "provenance": check_environment(),
    }


def rhs_time_control_cell(name, n, rtol):
    true = adaptive_cell(name, n, "S0", rtol, rhs_uses_time=True)
    false = adaptive_cell(name, n, "S0", rtol, rhs_uses_time=False)
    return {
        "problem": name,
        "N": n,
        "rtol": rtol,
        "true_counts": [true["accepted"], true["rejected"]],
        "false_counts": [false["accepted"], false["rejected"]],
        "attempts_equal": true["attempts"] == false["attempts"],
        "endpoint_equal": true["endpoint_sha256"] == false["endpoint_sha256"],
        "true_endpoint_sha256": true["endpoint_sha256"],
        "false_endpoint_sha256": false["endpoint_sha256"],
        "provenance": check_environment(),
    }


def fixed_cell(name, n, steps, graded=False):
    y0, ops, rhs, exact, final = setup(name, n)
    if graded:
        level = steps
        m1, m2 = 512 * 2**level, 4096 * 2**level
        t1 = 0.05 * (np.arange(m1 + 1) / m1) ** 4
        t2 = np.linspace(0.05, final, m2 + 1)
        h = jnp.asarray(np.diff(np.concatenate((t1, t2[1:]))))

        @jax.jit
        def solve():
            def body(carry, dt):
                state, t = carry
                return (_etdrk4_step_at(state, t, t + dt / 2, t + dt, dt, ops, rhs), t + dt), None

            return jax.lax.scan(body, (y0, jnp.asarray(0.0)), h)[0][0]

        n_steps = m1 + m2
        config = "F_g"
    else:
        dt = final / steps

        def solve():
            return etd_integrate(y0, (0.0, final), dt, ops, rhs, method="etdrk4", save_every=steps)[
                1
            ][-1]

        n_steps = steps
        config = "F"
    first = solve()
    jax.tree.map(jax.block_until_ready, first)
    start = time.perf_counter()
    state = solve()
    jax.tree.map(jax.block_until_ready, state)
    steady = time.perf_counter() - start
    finite = all(np.all(np.isfinite(np.asarray(v))) for v in state.values())
    row = {
        "problem": name,
        "N": n,
        "config": config,
        "steps": n_steps,
        "level": steps if graded else None,
        "attempts": n_steps,
        "accepted": n_steps if finite else None,
        "rejected": 0,
        "NFE": 4 * n_steps,
        "lean_NFE": 4 * n_steps,
        "transforms": 9 * n_steps * len(y0),
        "success": finite,
        "status": "finite" if finite else "non-finite",
        "endpoint_sha256": endpoint_sha(state),
        "steady_seconds": steady,
        "provenance": check_environment(),
    }
    if finite:
        row.update(error_summary(state, reference(name, n, exact, final), name, n))
    return row


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def stage_a():
    verify_references()
    print(json.dumps(check_environment()), flush=True)
    prior = [json.loads(line) for line in RAW.read_text().splitlines()] if RAW.exists() else []

    def record_key(r):
        value = (
            r["steps"] if r["config"] == "F" else r["level"] if r["config"] == "F_g" else r["rtol"]
        )
        return r["problem"], r["N"], r["config"], value

    done = {record_key(r) for r in prior}
    original = json.loads((EVIDENCE / "results/d0b.json").read_text())
    fixed_steps = {
        (r["problem"], r["N"]): [c["steps"] for c in r["cells"]]
        for r in original["B2"]["fixed_ladder"]
    }
    fixed_cost = {
        (r["problem"], r["N"], c["steps"]): c["wall_seconds"]
        for r in original["B2"]["fixed_ladder"]
        for c in r["cells"]
    }
    planning = {
        (r["problem"], r["N"], r["config"]): r
        for r in (
            json.loads(line)
            for line in (SCRATCH / "results/cost_gate.jsonl").read_text().splitlines()
        )
    }
    tasks = []
    for name, n in CASES:
        tasks.extend((name, n, "F", steps) for steps in fixed_steps[name, n])
        if name == "P5":
            tasks.extend((name, n, "F_g", level) for level in range(5))
        for config in CONFIGS:
            if name == "P1" and config in ("E0b", "X"):
                continue
            tasks.extend((name, n, config, rtol) for rtol in RTOLS)

    def projected_seconds(task):
        name, n, config, value = task
        if config == "F":
            return 2 * fixed_cost[name, n, value]
        if config == "F_g":
            return 2 * max(fixed_cost[name, n, s] for s in fixed_steps[name, n])
        old = planning.get((name, n, config))
        return min(900.0, old["first_seconds"] + old["steady_seconds"]) if old else 900.0

    def key(task):
        return task

    # E1 is chosen only after H2; reserve the slower pair until that choice exists.
    e1_reserve = 0.0
    for name, n in CASES:
        pair_costs = [
            planning[name, n, method]["first_seconds"] + planning[name, n, method]["steady_seconds"]
            for method in ("E0a", "E0b")
            if (name, n, method) in planning
        ]
        e1_reserve += 11 * 1.5 * (max(pair_costs) if pair_costs else 900.0)

    remaining = [task for task in tasks if key(task) not in done]
    spent = sum(row.get("wall_seconds_total", 0.0) for row in prior)
    initial_projection = spent + sum(projected_seconds(task) for task in remaining) + e1_reserve
    print(
        json.dumps(
            {
                "stage_a_initial_projection_hours": initial_projection / 3600,
                "remaining_records": len(remaining),
            }
        ),
        flush=True,
    )
    if initial_projection > 24 * 3600:
        raise RuntimeError("Stage A projected cost exceeds 24 h before starting")
    env = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    for name, n, config, value in tasks:
        if (name, n, config, value) in done:
            continue
        args = [sys.executable, str(Path(__file__)), "cell", name, str(n), config, str(value)]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                args, env=env, cwd=ROOT, capture_output=True, text=True, timeout=900
            )
            if proc.returncode:
                row = {
                    "problem": name,
                    "N": n,
                    "config": config,
                    "reason": proc.stderr[-2000:],
                    "status": "worker failed",
                }
            else:
                row = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            row = {
                "problem": name,
                "N": n,
                "config": config,
                "status": "not run (time)",
                "reason": "15-minute per-configuration guard",
            }
        row["wall_seconds_total"] = time.perf_counter() - started
        row["rtol" if config not in ("F", "F_g") else "level" if config == "F_g" else "steps"] = (
            value
        )
        append(RAW, row)
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "problem",
                        "N",
                        "config",
                        "rtol",
                        "steps",
                        "level",
                        "status",
                        "success",
                        "accepted",
                        "rejected",
                        "steady_seconds",
                        "wall_seconds_total",
                        "reason",
                    )
                    if k in row
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if row.get("status") == "worker failed" and any(
            marker in row.get("reason", "")
            for marker in ("moljax imported from", "D0b reference manifest mismatch")
        ):
            raise RuntimeError(row["reason"])
        spent += row["wall_seconds_total"]
        done.add((name, n, config, value))
        remaining = [task for task in tasks if task not in done]
        projected = spent + sum(projected_seconds(task) for task in remaining) + e1_reserve
        if projected > 24 * 3600:
            gate = {
                "status": "HOLD",
                "completed_records": len(done),
                "remaining_records": len(remaining),
                "spent_seconds": spent,
                "projected_remainder_seconds": projected - spent,
                "unselected_e1_reserve_seconds": e1_reserve,
                "projected_total_hours": projected / 3600,
                "planning_source": str(SCRATCH / "results/cost_gate.jsonl"),
            }
            (SCRATCH / "results/stage_a_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
            print(json.dumps(gate), flush=True)
            return False
    return True


def probes():
    verify_references()
    env = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    for name, n in CASES:
        path = SCRATCH / f"results/probe_{name}_{n}.json"
        if not path.exists():
            proc = subprocess.run(
                [sys.executable, str(Path(__file__)), "probe-cell", name, str(n)],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if proc.returncode:
                raise RuntimeError(f"estimator probe {name}/{n}: {proc.stderr[-2000:]}")
            path.write_text(proc.stdout.strip().splitlines()[-1] + "\n")
            print(f"estimator probe {name}/{n} complete", flush=True)
    for name, n in CASES:
        if name == "P1":
            continue
        for rtol in (1e-4, 1e-6):
            path = SCRATCH / f"results/rhs_control_{name}_{n}_{rtol:.0e}.json"
            if path.exists():
                continue
            proc = subprocess.run(
                [sys.executable, str(Path(__file__)), "control-cell", name, str(n), str(rtol)],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if proc.returncode:
                raise RuntimeError(f"rhs control {name}/{n}/{rtol}: {proc.stderr[-2000:]}")
            path.write_text(proc.stdout.strip().splitlines()[-1] + "\n")
            print(f"rhs control {name}/{n}/{rtol} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("cell", "probe-cell", "control-cell", "probes", "stage-a", "all")
    )
    parser.add_argument("problem", nargs="?")
    parser.add_argument("N", nargs="?", type=int)
    parser.add_argument("config", nargs="?")
    parser.add_argument("value", nargs="?", type=float)
    args = parser.parse_args()
    check_environment()
    if args.mode == "cell":
        verify_references()
        if args.config == "F":
            row = fixed_cell(args.problem, args.N, int(args.value))
        elif args.config == "F_g":
            row = fixed_cell(args.problem, args.N, int(args.value), graded=True)
        else:
            row = adaptive_cell(args.problem, args.N, args.config, args.value)
        print(json.dumps(row, sort_keys=True, allow_nan=False))
    elif args.mode == "probe-cell":
        print(
            json.dumps(estimator_power_cell(args.problem, args.N), sort_keys=True, allow_nan=False)
        )
    elif args.mode == "control-cell":
        rtol = args.value if args.value is not None else float(args.config)
        print(
            json.dumps(
                rhs_time_control_cell(args.problem, args.N, rtol), sort_keys=True, allow_nan=False
            )
        )
    elif args.mode == "probes":
        probes()
    elif args.mode in ("stage-a", "all"):
        probes()
        if stage_a() and args.mode == "all":
            from benchmarks.d3_work_precision_analysis import complete_study

            complete_study()


if __name__ == "__main__":
    main()
