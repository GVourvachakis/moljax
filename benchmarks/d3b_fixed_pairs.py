"""Exploratory, post hoc D3 decomposition using fixed-step embedded pairs.

Run ``JAX_PLATFORMS=cpu PYTHONPATH="$PWD" python benchmarks/d3b_fixed_pairs.py all``.
Checkpointed measurements and logs stay in ~/moljax_scratch/d3b/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from benchmarks import d3_work_precision as d3
from benchmarks import d3_work_precision_analysis as analysis

jax = d3.jax
jnp = d3.jnp
np = d3.np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path.home() / "moljax_scratch/d3b"
RAW = SCRATCH / "results/ladders.jsonl"
D3_JSON = ROOT / "benchmarks/results/d3_work_precision.json"
OUTPUT = ROOT / "benchmarks/results/d3b_fixed_pairs.json"
REPORT = ROOT / "benchmarks/results/D3B_SUMMARY.md"
TARGETS = analysis.TARGETS
PAIR = {"Z43": d3.ERK43ZB, "Z32": d3.ERK32ZB}
NFE_STEP = {"Z43": 5, "Z32": 4}
TIME_LIMIT = 600.0


def sources():
    d3.check_environment()
    d3.verify_references()
    d3_result = json.loads(D3_JSON.read_text())
    actual = hashlib.sha256(d3.RAW.read_bytes()).hexdigest()
    if actual != d3_result["raw_jsonl_sha256"]:
        raise RuntimeError("D3 stage_a.jsonl checksum mismatch")
    return d3_result


def fixed_steps(d3_result, name, n):
    return sorted(
        row["steps"]
        for row in d3_result["records"]
        if row["problem"] == name and row["N"] == n and row["config"] == "F"
    )


def solve_fixed(name, n, config, steps, graded=False):
    y0, ops, rhs, exact, final = d3.setup(name, n)
    tableau = PAIR[config]
    if graded:
        level = steps
        m1, m2 = 512 * 2**level, 4096 * 2**level
        t1 = 0.05 * (np.arange(m1 + 1) / m1) ** 4
        t2 = np.linspace(0.05, final, m2 + 1)
        boundaries = np.concatenate((t1, t2[1:]))
        times = jnp.asarray(boundaries[:-1])
        sizes = jnp.asarray(np.diff(boundaries))
        n_steps = m1 + m2

        @jax.jit
        def solve():
            def body(state, item):
                t, h = item
                stage_times = tuple(t + float(c) * h for c in tableau.c)
                high, _ = d3.pair_step_at(state, stage_times, h, ops, rhs, tableau)
                return high, None

            return jax.lax.scan(body, y0, (times, sizes))[0]

    else:
        n_steps = steps
        h = final / steps
        times = jnp.arange(steps, dtype=jnp.float64) * h

        @jax.jit
        def solve():
            def body(state, t):
                stage_times = tuple(t + float(c) * h for c in tableau.c)
                high, _ = d3.pair_step_at(state, stage_times, h, ops, rhs, tableau)
                return high, None

            return jax.lax.scan(body, y0, times)[0]

    return solve, n_steps, exact, final


def cell(name, n, config, steps, graded=False):
    solve, n_steps, exact, final = solve_fixed(name, n, config, steps, graded)
    start = time.perf_counter()
    warm = solve()
    jax.tree.map(jax.block_until_ready, warm)
    warm_seconds = time.perf_counter() - start
    if warm_seconds > TIME_LIMIT:
        return {
            "problem": name,
            "N": n,
            "config": config,
            "level": steps if graded else None,
            "steps": n_steps,
            "status": "not run (time)",
            "reason": "warmup exceeded 10 minutes",
            "warmup_seconds": warm_seconds,
        }
    start = time.perf_counter()
    state = solve()
    jax.tree.map(jax.block_until_ready, state)
    steady_seconds = time.perf_counter() - start
    if steady_seconds > TIME_LIMIT:
        return {
            "problem": name,
            "N": n,
            "config": config,
            "level": steps if graded else None,
            "steps": n_steps,
            "status": "not run (time)",
            "reason": "steady run exceeded 10 minutes",
            "warmup_seconds": warm_seconds,
            "steady_seconds": steady_seconds,
        }
    finite = all(np.isfinite(np.asarray(value)).all() for value in state.values())
    row = {
        "problem": name,
        "N": n,
        "config": config,
        "level": steps if graded else None,
        "steps": n_steps,
        "NFE": NFE_STEP[config] * n_steps,
        "NFE_with_FSAL": 3 * n_steps + 1 if config == "Z32" else None,
        "status": "finite" if finite else "non-finite",
        "success": finite,
        "endpoint_sha256": d3.endpoint_sha(state),
        "warmup_seconds": warm_seconds,
        "steady_seconds": steady_seconds,
    }
    if finite:
        row.update(d3.error_summary(state, d3.reference(name, n, exact, final), name, n))
    return row


def nfe_check():
    y0, ops, rhs, _, final = d3.setup("P2", 32)
    results = {}
    for config, tableau in PAIR.items():
        counter = [0]

        def counted(state, t):
            counter[0] += 1
            return rhs(state, t)

        h = final / 1024
        with jax.disable_jit():
            high, low = d3.pair_step_at(
                y0, tuple(float(c) * h for c in tableau.c), h, ops, counted, tableau
            )
            jax.tree.map(jax.block_until_ready, (high, low))
        if counter[0] != NFE_STEP[config]:
            raise RuntimeError(f"{config} NFE mismatch: {counter[0]}")
        results[config] = {"as_run_calls_per_step": counter[0], "stage_count": len(tableau.c)}
    return results


def spot_check(d3_result):
    checks = []
    for name, n in (("P2", 128), ("P5", 64)):
        for config in ("S0", "E0b"):
            expected = next(
                row
                for row in d3_result["records"]
                if row["problem"] == name
                and row["N"] == n
                and row["config"] == config
                and row.get("rtol") == 1e-4
            )
            actual = d3.adaptive_cell(name, n, config, 1e-4)
            equal = (actual["accepted"], actual["rejected"], actual["endpoint_sha256"]) == (
                expected["accepted"],
                expected["rejected"],
                expected["endpoint_sha256"],
            )
            checks.append(
                {
                    "problem": name,
                    "N": n,
                    "config": config,
                    "expected_counts": [expected["accepted"], expected["rejected"]],
                    "actual_counts": [actual["accepted"], actual["rejected"]],
                    "expected_sha256": expected["endpoint_sha256"],
                    "actual_sha256": actual["endpoint_sha256"],
                    "bitwise_match": equal,
                }
            )
            if not equal:
                raise RuntimeError(f"D3 bitwise spot-check mismatch: {name}/{n}/{config}")
    return checks


def record_id(name, n, config, value, graded):
    return name, n, config, value, graded


def run_ladders(d3_result):
    RAW.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in RAW.read_text().splitlines()] if RAW.exists() else []
    rows = {
        record_id(
            r["problem"],
            r["N"],
            r["config"],
            r["level"] if r["level"] is not None else r["steps"],
            r["level"] is not None,
        ): r
        for r in existing
    }
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))

    def measure(name, n, config, value, graded=False):
        key = record_id(name, n, config, value, graded)
        if key in rows:
            return rows[key]
        command = [sys.executable, str(Path(__file__)), "cell", name, str(n), config, str(value)]
        if graded:
            command.append("--graded")
        try:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=2 * TIME_LIMIT + 90,
            )
        except subprocess.TimeoutExpired:
            row = {
                "problem": name,
                "N": n,
                "config": config,
                "level": value if graded else None,
                "steps": (512 + 4096) * 2**value if graded else value,
                "status": "not run (time)",
                "reason": "cell process exceeded two 10-minute runs",
            }
        else:
            if proc.returncode:
                raise RuntimeError(f"{name}/{n}/{config}/{value}: {proc.stderr[-2000:]}")
            row = json.loads(proc.stdout.strip().splitlines()[-1])
        d3.append(RAW, row)
        rows[key] = row
        print(
            json.dumps(
                {
                    "case": f"{name}/{n}",
                    "config": config,
                    "value": value,
                    "status": row["status"],
                    "relative_error": row.get("error_inf_relative"),
                    "steady_seconds": row.get("steady_seconds"),
                }
            ),
            flush=True,
        )
        return row

    for name, n in d3.CASES:
        for config in ("Z43", "Z32"):
            if name == "P1" and config == "Z43":
                continue
            steps = fixed_steps(d3_result, name, n)
            for count in steps:
                last = measure(name, n, config, count)
                if last["status"] == "not run (time)":
                    break
            if last["status"] == "not run (time)":
                continue
            count = steps[-1]
            while True:
                last = measure(name, n, config, count)
                if last["status"] == "not run (time)" or (
                    last["success"] and last["error_inf_relative"] <= 1e-6
                ):
                    break
                count *= 2
                measure(name, n, config, count)
        if name == "P5":
            for level in range(5):
                measure(name, n, "Z43", level, graded=True)
    return list(rows.values())


def slim_f(row):
    keys = (
        "problem",
        "N",
        "config",
        "steps",
        "level",
        "NFE",
        "status",
        "success",
        "error_inf_absolute",
        "error_inf_per_field",
        "error_inf_relative",
        "error_inf_relative_per_field",
        "validity",
        "endpoint_sha256",
        "steady_seconds",
    )
    return {key: row.get(key) for key in keys}


def add_orders(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["problem"], row["N"], row["config"]].append(row)
    for case_rows in grouped.values():
        case_rows.sort(key=lambda r: r["steps"])
        prior = None
        for row in case_rows:
            error = row.get("error_inf_relative")
            old = prior.get("error_inf_relative") if prior else None
            row["observed_order"] = (
                math.log2(old / error)
                if old is not None
                and error is not None
                and old > 0
                and error > 0
                and row["steps"] == 2 * prior["steps"]
                else None
            )
            prior = row


def decomposition(d3_result, ladders):
    old = analysis.groups(d3_result["records"])
    fresh = analysis.groups(ladders)
    outputs, missing = [], []
    for name, n in d3.CASES:
        variants = [("E0a", "Z32", "F")]
        if name != "P1":
            variants.append(("E0b", "Z43", "F"))
        if name == "P5":
            variants.append(("E0b", "Z43_g", "F_g"))
        for candidate, z_config, f_config in variants:
            for target in TARGETS:
                values = {
                    "S0": analysis.interpolate(old[name, n, "S0"], target),
                    f_config: analysis.interpolate(old[name, n, f_config], target),
                    z_config: analysis.interpolate(fresh[name, n, z_config], target),
                    candidate: analysis.interpolate(old[name, n, candidate], target),
                }
                unavailable = []
                for config, value in values.items():
                    if value is None:
                        records = (
                            old[name, n, config]
                            if config in ("S0", "F", "F_g", "E0a", "E0b")
                            else fresh[name, n, config]
                        )
                        usable = analysis.pareto(records)
                        reason = (
                            (
                                f"outside usable Pareto error range "
                                f"[{usable[0]['error_inf_relative']:.6g}, "
                                f"{usable[-1]['error_inf_relative']:.6g}]"
                            )
                            if usable
                            else "no usable finite rung"
                        )
                        unavailable.append({"config": config, "reason": reason})
                base = {
                    "problem": name,
                    "N": n,
                    "target": target,
                    "candidate": candidate,
                    "fixed_pair": z_config,
                    "fixed_baseline": f_config,
                }
                if unavailable:
                    missing.append({**base, "unavailable": unavailable})
                    continue
                s0, f, z, e = (values[k] for k in ("S0", f_config, z_config, candidate))
                factors = {"adaptivity": s0 / f, "method": f / z, "control": z / e}
                product = math.prod(factors.values())
                d3_q = next(
                    cell["Q_vs_S0_NFE"]
                    for cell in d3_result["matched_targets"][f"{name}/{n}/{candidate}"]
                    if cell["target"] == target
                )
                discrepancy = product - d3_q
                if not math.isclose(product, d3_q, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError(f"decomposition identity mismatch: {base}, {discrepancy}")
                outputs.append(
                    {
                        **base,
                        "NFE": values,
                        "factors": factors,
                        "product": product,
                        "D3_Q": d3_q,
                        "product_minus_D3_Q": discrepancy,
                    }
                )
    return outputs, missing


def render(payload):
    lines = [
        "# D3b fixed-pair decomposition — exploratory, post hoc",
        "",
        "This analysis was formed after seeing D3. It changes no D3 verdict.",
        "",
        "## Provenance and checks",
        "",
        f"- Base `{payload['provenance']['head']}`; Python {payload['provenance']['python']}; "
        f"JAX {payload['provenance']['jax']}; backend {payload['provenance']['backend']}.",
        "Source: `d3b_fixed_pairs.json`, `/provenance`.",
        "",
        f"- As-run NFE per Z43/Z32 step: "
        f"{payload['nfe_check']['Z43']['as_run_calls_per_step']}/"
        f"{payload['nfe_check']['Z32']['as_run_calls_per_step']}; "
        f"bitwise D3 spot checks: {sum(x['bitwise_match'] for x in payload['spot_checks'])}/"
        f"{len(payload['spot_checks'])}.",
        "Source: `d3b_fixed_pairs.json`, `/nfe_check` and `/spot_checks`.",
        "",
        "## Fixed-step ladders",
        "",
        "Z32's FSAL NFE is hypothetical; the as-run stepper makes four calls per step. "
        "A dash for order means the adjacent error ratio was unavailable.",
        "",
        "| Case | Ladder | Steps | NFE | FSAL NFE | Relative error | Order | Validity/status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["ladders"]:
        case = f"{row['problem']}/{row['N']}"
        if row.get("level") is not None:
            case += f" level {row['level']}"
        lines.append(
            f"| {case} | {row['config']} | {row['steps']} | {row.get('NFE') or '—'} | "
            f"{row.get('NFE_with_FSAL') or '—'} | "
            f"{row['error_inf_relative']:.6g} | "
            if row.get("error_inf_relative") is not None
            else f"| {case} | {row['config']} | {row['steps']} | {row.get('NFE') or '—'} | "
            f"{row.get('NFE_with_FSAL') or '—'} | — | "
        )
        lines[-1] += f"{row['observed_order']:.3g}" if row["observed_order"] is not None else "—"
        lines[-1] += f" | {row.get('validity') or row['status']} |"
    lines += [
        "Source: `d3b_fixed_pairs.json`, `/ladders` (every row).",
        "",
        "## Decomposition at matched accuracy",
        "",
        "| Case | Pair | Baseline | Target | Adaptivity | Method | Control | Product | D3 Q | Difference |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["decomposition"]:
        a, m, c = (row["factors"][key] for key in ("adaptivity", "method", "control"))
        lines.append(
            f"| {row['problem']}/{row['N']} | {row['candidate']} | {row['fixed_baseline']} | "
            f"{row['target']:.0e} | {a:.5g} | {m:.5g} | {c:.5g} | "
            f"{row['product']:.5g} | {row['D3_Q']:.5g} | "
            f"{row['product_minus_D3_Q']:.2g} |"
        )
    lines += [
        "Source: `d3b_fixed_pairs.json`, `/decomposition` (every row).",
        "",
        "## Missing matched-target cells",
        "",
        "| Case | Pair | Baseline | Target | Unavailable factor inputs and reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in payload["missing"]:
        reason = "; ".join(f"{x['config']}: {x['reason']}" for x in row["unavailable"])
        lines.append(
            f"| {row['problem']}/{row['N']} | {row['candidate']} | "
            f"{row['fixed_baseline']} | {row['target']:.0e} | {reason} |"
        )
    lines += ["Source: `d3b_fixed_pairs.json`, `/missing` (every row).", ""]
    return "\n".join(lines)


def finalise(d3_result, nfe, spots, measured):
    old = [slim_f(r) for r in d3_result["records"] if r["config"] in ("F", "F_g")]
    fresh = [
        dict(r, config="Z43_g" if r["config"] == "Z43" and r["level"] is not None else r["config"])
        for r in measured
    ]
    ladders = old + fresh
    add_orders(ladders)
    decomposed, missing = decomposition(d3_result, ladders)
    payload = {
        "label": "exploratory, post hoc; changes no D3 verdict",
        "provenance": {
            **d3.check_environment(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "d3_json_sha256": hashlib.sha256(D3_JSON.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "settings": {
            "targets": TARGETS,
            "extension_error_threshold": 1e-6,
            "single_run_time_limit_seconds": TIME_LIMIT,
            "fixed_steps_source": "D0b B2, via D3 F rows",
            "graded_levels": list(range(5)),
            "metric": "max per-field relative infinity error",
            "validity": "D0b absolute-error rule via D3 error_summary",
        },
        "nfe_check": nfe,
        "spot_checks": spots,
        "ladders": ladders,
        "decomposition": decomposed,
        "missing": missing,
        "raw_jsonl_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
    }
    OUTPUT.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if OUTPUT.stat().st_size > 1_000_000:
        raise RuntimeError(f"D3b JSON exceeds 1 MB: {OUTPUT.stat().st_size}")
    REPORT.write_text(render(json.loads(OUTPUT.read_text())))
    print(
        f"D3b complete: {len(ladders)} ladder rows, {len(decomposed)} decompositions, "
        f"{len(missing)} missing; {OUTPUT.stat().st_size} JSON bytes",
        flush=True,
    )
    files = []
    for path in sorted(SCRATCH.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts:
            files.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(SCRATCH)}"
            )
    (SCRATCH / "results/SHA256SUMS").write_text("\n".join(files) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "cell", "nfe", "spot"))
    parser.add_argument("problem", nargs="?")
    parser.add_argument("N", nargs="?", type=int)
    parser.add_argument("config", nargs="?")
    parser.add_argument("value", nargs="?", type=int)
    parser.add_argument("--graded", action="store_true")
    args = parser.parse_args()
    baseline = sources()
    if args.mode == "cell":
        print(
            json.dumps(
                cell(args.problem, args.N, args.config, args.value, args.graded),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return
    if args.mode == "nfe":
        print(json.dumps(nfe_check(), sort_keys=True))
        return
    if args.mode == "spot":
        print(json.dumps(spot_check(baseline), sort_keys=True))
        return
    SCRATCH.joinpath("results").mkdir(parents=True, exist_ok=True)
    nfe = nfe_check()
    spots = spot_check(baseline)
    (SCRATCH / "results/gates.json").write_text(
        json.dumps({"nfe": nfe, "spot_checks": spots}, sort_keys=True, indent=2) + "\n"
    )
    measured = run_ladders(baseline)
    finalise(baseline, nfe, spots, measured)


if __name__ == "__main__":
    main()
