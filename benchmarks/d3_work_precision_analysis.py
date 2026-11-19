"""Matched-accuracy and reporting analysis for the self-contained D3 study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks import d3_work_precision as d3

TARGETS = (1e-3, 1e-4, 1e-5, 1e-6)
TIMING_TARGET = 1e-4


def usable(row):
    return (
        row.get("success") is True
        and row.get("validity") != "indeterminate"
        and 0 < row.get("error_inf_relative", math.inf) <= 1e-2
        and row.get("NFE") is not None
    )


def pareto(rows, metric="NFE", rtol_range=None):
    """Keep points not dominated by a no-costlier point of smaller error."""
    eligible = [
        r
        for r in rows
        if usable(r)
        and r.get(metric) is not None
        and (
            rtol_range is None
            or r.get("rtol") is not None
            and rtol_range[0] <= r["rtol"] <= rtol_range[1]
        )
    ]
    eligible.sort(key=lambda r: (r["error_inf_relative"], r[metric]))
    frontier, best = [], math.inf
    for row in eligible:
        if row[metric] < best:
            frontier.append(row)
            best = row[metric]
    return frontier


def interpolate(rows, target, metric="NFE", rtol_range=None):
    frontier = pareto(rows, metric, rtol_range)
    if (
        not frontier
        or target < frontier[0]["error_inf_relative"]
        or target > frontier[-1]["error_inf_relative"]
    ):
        return None
    for row in frontier:
        if row["error_inf_relative"] == target:
            return float(row[metric])
    for low, high in zip(frontier, frontier[1:], strict=False):
        e0, e1 = low["error_inf_relative"], high["error_inf_relative"]
        if e0 <= target <= e1:
            fraction = math.log(target / e0) / math.log(e1 / e0)
            return math.exp(
                math.log(low[metric]) + fraction * (math.log(high[metric]) - math.log(low[metric]))
            )
    return None


def groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["problem"], row["N"], row["config"]].append(row)
    return result


def q_value(grouped, case, a, b, target, metric="NFE", rtol_range=None):
    name, n = case
    cost_a = interpolate(grouped[name, n, a], target, metric, rtol_range)
    cost_b = interpolate(grouped[name, n, b], target, metric, rtol_range)
    return None if cost_a is None or cost_b is None else cost_b / cost_a


def bracket(rows, target=TIMING_TARGET):
    frontier = pareto(rows)
    if (
        len(frontier) < 2
        or target < frontier[0]["error_inf_relative"]
        or target > frontier[-1]["error_inf_relative"]
    ):
        return None
    for low, high in zip(frontier, frontier[1:], strict=False):
        if low["error_inf_relative"] <= target <= high["error_inf_relative"]:
            return low, high
    return None


def record_id(row):
    value = (
        row["steps"]
        if row["config"] == "F"
        else row["level"]
        if row["config"] == "F_g"
        else row["rtol"]
    )
    return f"{row['problem']}_{row['N']}_{row['config']}_{value}"


def read_rows(work_dir):
    return [json.loads(line) for line in (work_dir / "stage_a.jsonl").read_text().splitlines()]


def matched_tables(rows, confirmation):
    grouped = groups(rows)
    tables = {}
    for name, n in d3.CASES:
        epsilon = confirmation[f"{name}_{n}"]["epsilon_conf"]
        for method in ("S0", "sensitivity", "E0a", "E0b"):
            if not grouped[name, n, method]:
                continue
            cells = []
            for target in TARGETS:
                entry = {"target": target, "confirmed": target >= 10 * epsilon}
                for metric in ("NFE", "lean_NFE"):
                    entry[metric] = interpolate(grouped[name, n, method], target, metric)
                    entry[f"Q_vs_S0_{metric}"] = (
                        None
                        if method == "S0"
                        else q_value(grouped, (name, n), method, "S0", target, metric)
                    )
                    entry[f"R_vs_F_{metric}"] = q_value(
                        grouped, (name, n), method, "F", target, metric
                    )
                    if name == "P5":
                        entry[f"R_vs_Fg_{metric}"] = q_value(
                            grouped, (name, n), method, "F_g", target, metric
                        )
                cells.append(entry)
            tables[f"{name}/{n}/{method}"] = cells
    return tables


def hypotheses(rows, confirmation):
    grouped = groups(rows)
    h1 = {}
    for method in ("E0a", "E0b"):
        by_problem = {}
        for name in ("P1", "P2", "P3", "P4", "P5"):
            if name == "P1" and method == "E0b":
                continue
            per_target = []
            for case in d3.CASES:
                if case[0] != name:
                    continue
                for target in TARGETS:
                    value = q_value(grouped, case, method, "S0", target)
                    if value is not None:
                        per_target.append({"N": case[1], "target": target, "Q": value})
            by_problem[name] = {
                "per_target": per_target,
                "median_Q": statistics.median(x["Q"] for x in per_target) if per_target else None,
            }
        positive = sum(
            x["median_Q"] is not None and x["median_Q"] >= 1.15 for x in by_problem.values()
        )
        loses = any(x["median_Q"] is not None and x["median_Q"] < 0.90 for x in by_problem.values())
        h1[method] = {
            "by_problem": by_problem,
            "positive_problem_count": positive,
            "any_problem_below_0p90": loses,
            "supported": positive >= 3 and not loses,
            "eligible_problem_count": sum(x["median_Q"] is not None for x in by_problem.values()),
        }
    h3 = {}
    for name, n in d3.CASES:
        for method in ("S0", "E0a", "E0b"):
            if not grouped[name, n, method]:
                continue
            for baseline in ("F", "F_g") if name == "P5" else ("F",):
                points = []
                for target in TARGETS:
                    q = q_value(grouped, (name, n), method, baseline, target)
                    if q is not None:
                        points.append({"target": target, "R": q})
                h3[f"{name}/{n}/{method}/{baseline}"] = {
                    "per_target": points,
                    "median_R": statistics.median(x["R"] for x in points) if points else None,
                }
    lean = {}
    confirmed = {}
    for method in ("E0a", "E0b"):
        lean[method], confirmed[method] = {}, {}
        for name, n in d3.CASES:
            points = []
            confirmed_points = []
            for target in TARGETS:
                q = q_value(grouped, (name, n), method, "S0", target, "lean_NFE")
                if q is not None:
                    points.append({"target": target, "Q": q})
                q = q_value(grouped, (name, n), method, "S0", target)
                if q is not None and target >= 10 * confirmation[f"{name}_{n}"]["epsilon_conf"]:
                    confirmed_points.append({"target": target, "Q": q})
            lean[method][f"{name}/{n}"] = points
            confirmed[method][f"{name}/{n}"] = confirmed_points
    return {
        "H1": h1,
        "H3": h3,
        "lean_NFE_Q": lean,
        "confirmed_target_Q": confirmed,
        "negative_result_clause": {
            "D2_not_proposed_upstream": not any(v["supported"] for v in h1.values())
        },
        "H3_structural_ceilings": {"step_doubling": 2 / 3, "ERK32ZB": 3 / 4, "ERK43ZB": 1},
    }


def tolerance_proportionality(rows):
    out = {}
    for name, n in d3.CASES:
        for config in ("S0", "sensitivity", "E0a", "E0b"):
            eligible = [
                r
                for r in rows
                if (r["problem"], r["N"], r["config"]) == (name, n, config)
                and usable(r)
                and r.get("rtol") is not None
            ]
            if len(eligible) >= 2:
                slope = float(
                    np.polyfit(
                        np.log([r["rtol"] for r in eligible]),
                        np.log([r["error_inf_relative"] for r in eligible]),
                        1,
                    )[0]
                )
                out[f"{name}/{n}/{config}"] = {"slope": slope, "n": len(eligible)}
    return out


def status(rows):
    counts = defaultdict(lambda: defaultdict(int))
    failures = []
    exclusions = defaultdict(int)
    for row in rows:
        key = f"{row['problem']}/{row['N']}"
        counts[key][row["config"]] += 1
        if row.get("success") is False or row.get("status") in ("worker failed", "not run (time)"):
            failures.append(
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
                        "reason",
                    )
                }
            )
        if row.get("validity") == "indeterminate" or row.get("error_inf_relative", 0) > 1e-2:
            exclusions[f"{key}/{row['config']}"] += 1
    return {
        "counts": {k: dict(v) for k, v in counts.items()},
        "failures": failures,
        "excluded_counts": dict(exclusions),
    }


def determinism(rows, work_dir, strict=False):
    grouped = groups(rows)
    requested = (
        ("P1", 64, "F", 32),
        ("P5", 64, "F_g", 0),
        ("P1", 64, "S0", 1e-4),
        ("P1", 64, "sensitivity", 1e-4),
        ("P1", 64, "E0a", 1e-4),
        ("P2", 128, "E0b", 1e-4),
    )
    out = []
    for name, n, config, value in requested:
        source = next(
            r
            for r in grouped[name, n, config]
            if (r["steps"] if config == "F" else r["level"] if config == "F_g" else r["rtol"])
            == value
        )
        proc = d3._worker("cell", name, n, config, value, strict)
        if proc.returncode:
            raise RuntimeError(
                f"determinism replay failed: {name}/{n}/{config}: {proc.stderr[-2000:]}"
            )
        replay = json.loads(proc.stdout.strip().splitlines()[-1])
        same = all(
            source.get(key) == replay.get(key)
            for key in ("accepted", "rejected", "endpoint_sha256")
        )
        out.append(
            {
                "case": f"{name}/{n}",
                "config": config,
                "value": value,
                "counts_and_endpoint_match": same,
                "endpoint_sha256": replay["endpoint_sha256"],
            }
        )
        if not same:
            raise RuntimeError(f"determinism mismatch: {name}/{n}/{config}")
    (work_dir / "determinism.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def timing_plan(rows):
    grouped = groups(rows)
    selected, no_bracket = [], []
    for name, n in d3.CASES:
        for config in ("F", "S0", "E0a", "E0b"):
            if not grouped[name, n, config]:
                no_bracket.append(
                    {"case": f"{name}/{n}", "config": config, "reason": "method ineligible"}
                )
                continue
            pair = bracket(grouped[name, n, config])
            if pair is None:
                no_bracket.append(
                    {
                        "case": f"{name}/{n}",
                        "config": config,
                        "reason": "target outside valid Pareto error range",
                    }
                )
                continue
            for row in pair:
                selected.append(
                    {
                        "id": record_id(row),
                        "problem": name,
                        "N": n,
                        "config": config,
                        "value": row["steps"] if config == "F" else row["rtol"],
                        "relative_error": row["error_inf_relative"],
                        "steady_seconds": row["steady_seconds"],
                    }
                )
    full_seconds = sum(50 * r["steady_seconds"] for r in selected)
    fallback = full_seconds > 8 * 3600
    for entry in selected:
        entry["runs_per_block"] = 4 if fallback and entry["steady_seconds"] > 5 else 10
    priced = sum(5 * r["runs_per_block"] * r["steady_seconds"] for r in selected)
    return {
        "target": TIMING_TARGET,
        "selected": selected,
        "no_bracket": no_bracket,
        "full_5x10_seconds": full_seconds,
        "fallback_applied": fallback,
        "priced_seconds": priced,
        "passes_8h_gate": priced <= 8 * 3600,
    }


def timing_callable(row):
    name, n, config = row["problem"], row["N"], row["config"]
    y0, ops, rhs, _, final = d3.setup(name, n)
    if config == "F":
        steps = row["steps"]
        return lambda: d3.etd_integrate(
            y0, (0.0, final), final / steps, ops, rhs, method="etdrk4", save_every=steps
        )[1][-1]
    method = d3.METHODS.get(config, "etdrk4")
    pid = d3.PIDParams(atol=1e-3 * row["rtol"], rtol=row["rtol"], dt_max=final)
    return lambda: (
        d3.etd_integrate_adaptive(
            y0,
            (0.0, final),
            1e-4 * final,
            ops,
            rhs,
            method=method,
            rhs_uses_time=method != "erk43zb",
            pid_params=pid,
            max_steps=row["max_steps"],
            save_every=row["save_every"],
        ).y_final
    )


def timing_case(rows, plan, work_dir, name, n):
    selected = [e for e in plan["selected"] if (e["problem"], e["N"]) == (name, n)]
    if not selected:
        return
    raw_path = work_dir / "timing.jsonl"
    existing = (
        {
            (r["id"], r["block"], r["repetition"])
            for r in (json.loads(line) for line in raw_path.read_text().splitlines())
        }
        if raw_path.exists()
        else set()
    )
    indexed = {record_id(r): r for r in rows}
    functions = {}
    for entry in selected:
        source = indexed[entry["id"]]
        solve = timing_callable(source)
        state = solve()
        d3.jax.tree.map(d3.jax.block_until_ready, state)
        if d3.endpoint_sha(state) != source["endpoint_sha256"]:
            raise RuntimeError(f"timing warmup endpoint mismatch: {entry['id']}")
        functions[entry["id"]] = solve
    for block in range(5):
        schedule = [
            (entry["id"], rep) for entry in selected for rep in range(entry["runs_per_block"])
        ]
        random.Random(20261116 + 1000 * n + 100 * block + ord(name[1])).shuffle(schedule)
        for identifier, rep in schedule:
            if (identifier, block, rep) in existing:
                continue
            start = time.perf_counter()
            state = functions[identifier]()
            d3.jax.tree.map(d3.jax.block_until_ready, state)
            d3.append(
                raw_path,
                {
                    "id": identifier,
                    "problem": name,
                    "N": n,
                    "block": block,
                    "repetition": rep,
                    "seconds": time.perf_counter() - start,
                },
            )
        print(json.dumps({"timing_case": f"{name}/{n}", "completed_block": block}), flush=True)


def run_timing(rows, plan, work_dir):
    if not plan["passes_8h_gate"]:
        raise RuntimeError("timing subset exceeds eight-hour gate")
    for name, n in d3.CASES:
        if not any((entry["problem"], entry["N"]) == (name, n) for entry in plan["selected"]):
            continue
        env = dict(os.environ, PYTHONPATH=str(d3.ROOT), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "timing-case", name, str(n), str(work_dir)],
            cwd=d3.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=8 * 3600,
        )
        if proc.returncode:
            raise RuntimeError(f"timing case {name}/{n} failed: {proc.stderr[-2000:]}")
        print(proc.stdout, flush=True)


def timed_summary(plan, rows, work_dir):
    raw = [json.loads(line) for line in (work_dir / "timing.jsonl").read_text().splitlines()]
    by_id = defaultdict(list)
    for r in raw:
        by_id[r["id"]].append(r["seconds"])
    summary = {}
    for entry in plan["selected"]:
        identifier = entry["id"]
        values = np.asarray(by_id[identifier], dtype=float)
        expected = 5 * entry["runs_per_block"]
        if len(values) != expected:
            raise RuntimeError(
                f"incomplete timing record {identifier}: {len(values)} != {expected}"
            )
        summary[identifier] = {
            "median_seconds": float(np.median(values)),
            "IQR_seconds": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "n_runs": len(values),
        }
    grouped = groups(rows)
    interpolated = {}
    for name, n in d3.CASES:
        for config in ("F", "S0", "E0a", "E0b"):
            pair = bracket(grouped[name, n, config])
            if pair is None:
                continue
            low, high = pair
            e0, e1 = low["error_inf_relative"], high["error_inf_relative"]
            fraction = math.log(TIMING_TARGET / e0) / math.log(e1 / e0)
            ids = (record_id(low), record_id(high))
            value = math.exp(
                math.log(summary[ids[0]]["median_seconds"])
                + fraction
                * (
                    math.log(summary[ids[1]]["median_seconds"])
                    - math.log(summary[ids[0]]["median_seconds"])
                )
            )
            interpolated[f"{name}/{n}/{config}"] = {
                "seconds": value,
                "fraction": fraction,
                "record_ids": ids,
            }
    rng = np.random.default_rng(20261116)
    ci = {}
    for key, entry in interpolated.items():
        name, n, config = key.split("/")
        baseline = interpolated.get(f"{name}/{n}/S0")
        if baseline is None or config == "S0":
            continue
        ratios = []
        for _ in range(2000):

            def sampled(interp):
                medians = [
                    float(
                        np.median(
                            rng.choice(by_id[identifier], len(by_id[identifier]), replace=True)
                        )
                    )
                    for identifier in interp["record_ids"]
                ]
                return math.exp(
                    math.log(medians[0])
                    + interp["fraction"] * (math.log(medians[1]) - math.log(medians[0]))
                )

            ratios.append(sampled(baseline) / sampled(entry))
        ci[key] = {
            "Q_time_vs_S0": baseline["seconds"] / entry["seconds"],
            "bootstrap_95pct": [float(x) for x in np.quantile(ratios, [0.025, 0.975])],
        }
    return {
        "records": summary,
        "interpolated": interpolated,
        "ratios": ci,
        "bootstrap_seed": 20261116,
        "bootstrap_resamples": 2000,
    }


def _fmt(value, pointer, digits=4):
    return (
        f"{value:.{digits}g} (`{pointer}`)" if value is not None else f"unattainable (`{pointer}`)"
    )


def render(payload):
    lines = [
        "# D3: adaptive exponential work precision",
        "",
        "## Setup and protocol",
        "",
        "The H1 threshold and matched accuracy targets were fixed before comparison. "
        "The reference arrays and their checksums are in `d3_references.npz` and "
        "`d3_references.json`. CPU timings are secondary evidence.",
        "",
        f"H1 requires Q ≥ {_fmt(payload['settings']['H1_thresholds']['support_Q'], '/settings/H1_thresholds/support_Q')} "
        f"on at least {_fmt(payload['settings']['H1_thresholds']['minimum_problem_count'], '/settings/H1_thresholds/minimum_problem_count')} "
        f"eligible problems, with no problem below {_fmt(payload['settings']['H1_thresholds']['loss_Q'], '/settings/H1_thresholds/loss_Q')}.",
        "",
        "| Case | Reference confirmation level | Method |",
        "| --- | ---: | --- |",
    ]
    for case, detail in payload["reference_confirmation"].items():
        lines.append(
            f"| {case} | {_fmt(detail['epsilon_conf'], '/reference_confirmation/' + case + '/epsilon_conf')} | {detail['method']} |"
        )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/settings`, `/reference_confirmation`, `/provenance`.",
        "",
        "## H1: embedded pairs against step doubling",
        "",
        "| Pair | Problem | Median Q | Supported |",
        "| --- | --- | ---: | --- |",
    ]
    for method, result in payload["hypotheses"]["H1"].items():
        for problem, cell in result["by_problem"].items():
            ptr = f"/hypotheses/H1/{method}/by_problem/{problem}/median_Q"
            lines.append(
                f"| {method} | {problem} | {_fmt(cell['median_Q'], ptr)} | {result['supported']} |"
            )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/hypotheses/H1`.",
        "",
        "The primary Q uses nominal NFE. Lean NFE and CPU-time ratios are reported as secondary checks.",
        "",
        "| Case | Pair | Target | Q | Lean Q | Confirmed |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for key, cells in payload["matched_targets"].items():
        if not key.endswith(("/E0a", "/E0b")):
            continue
        for index, cell in enumerate(cells):
            ptr = f"/matched_targets/{key}/{index}"
            lines.append(
                f"| {key.rsplit('/', 1)[0]} | {key.rsplit('/', 1)[1]} | "
                f"{_fmt(cell['target'], ptr + '/target')} | "
                f"{_fmt(cell['Q_vs_S0_NFE'], ptr + '/Q_vs_S0_NFE')} | "
                f"{_fmt(cell['Q_vs_S0_lean_NFE'], ptr + '/Q_vs_S0_lean_NFE')} | {cell['confirmed']} |"
            )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/matched_targets`.",
        "",
        "## H3: fixed-step anchors (descriptive)",
        "",
        "| Case / method / baseline | Median R |",
        "| --- | ---: |",
    ]
    for key, cell in payload["hypotheses"]["H3"].items():
        lines.append(f"| {key} | {_fmt(cell['median_R'], '/hypotheses/H3/' + key + '/median_R')} |")
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/hypotheses/H3`.",
        "",
        "## Estimator power and tolerance proportionality",
        "",
        "| Case | Method | Time fraction | Median slope | Nominal power |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for case_index, probe in enumerate(payload["estimator_power"]):
        for entry_index, entry in enumerate(probe["entries"]):
            ptr = f"/estimator_power/{case_index}/entries/{entry_index}"
            lines.append(
                f"| {probe['problem']}/{probe['N']} | {entry['method']} | "
                f"{_fmt(entry['target_fraction'], ptr + '/target_fraction')} | "
                f"{_fmt(entry['median_slope_k_ge_2'], ptr + '/median_slope_k_ge_2')} | "
                f"{_fmt(entry['nominal_power'], ptr + '/nominal_power')} |"
            )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/estimator_power`.",
        "",
        "| Case / method | Tolerance slope | Valid records |",
        "| --- | ---: | ---: |",
    ]
    for key, cell in payload["tolerance_proportionality"].items():
        lines.append(
            f"| {key} | {_fmt(cell['slope'], '/tolerance_proportionality/' + key + '/slope')} | "
            f"{_fmt(cell['n'], '/tolerance_proportionality/' + key + '/n')} |"
        )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/tolerance_proportionality`.",
        "",
        "## CPU timing ratios",
        "",
        "| Case / method | Time ratio against S0 | Bootstrap interval |",
        "| --- | ---: | ---: |",
    ]
    for key, cell in payload["timing"]["ratios"].items():
        ptr = "/timing/ratios/" + key
        lo, hi = cell["bootstrap_95pct"]
        lines.append(
            f"| {key} | {_fmt(cell['Q_time_vs_S0'], ptr + '/Q_time_vs_S0')} | "
            f"{_fmt(lo, ptr + '/bootstrap_95pct/0')} to {_fmt(hi, ptr + '/bootstrap_95pct/1')} |"
        )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/timing`.",
        "",
        "## Limits",
        "",
        "ERK43ZB's 1/6 stage is outside the tick clock's one-rounding guarantee, so it is "
        "restricted to autonomous RHS. Measurements used a laptop CPU. Filter presets and the "
        "adaptive IMEX comparator remain on [experiment/etd-embedded-pairs](https://github.com/GVourvachakis/moljax/tree/experiment/etd-embedded-pairs); they are not part of this branch.",
        "",
        "Source: `d3_work_precision.json`, `/settings`.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(rows, matched):
    result_dir = d3.ROOT / "benchmarks/results"
    record_columns = (
        "problem",
        "N",
        "config",
        "rtol",
        "steps",
        "level",
        "accepted",
        "rejected",
        "attempts",
        "NFE",
        "lean_NFE",
        "transforms",
        "error_inf_relative",
        "validity",
        "status",
        "endpoint_sha256",
    )
    with (result_dir / "d3_records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=record_columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in record_columns})
    with (result_dir / "d3_matched_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "problem",
                "N",
                "config",
                "target",
                "confirmed",
                "Q_vs_S0_NFE",
                "Q_vs_S0_lean_NFE",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for key, cells in matched.items():
            name, n, config = key.split("/")
            if config not in ("E0a", "E0b"):
                continue
            for cell in cells:
                writer.writerow(
                    {
                        "problem": name,
                        "N": n,
                        "config": config,
                        **{
                            field: cell[field]
                            for field in ("target", "confirmed", "Q_vs_S0_NFE", "Q_vs_S0_lean_NFE")
                        },
                    }
                )


def complete_study(work_dir, strict=False):
    d3.WORK_DIR = work_dir
    d3.verify_references()
    rows = read_rows(work_dir)
    expected = len(list(d3._tasks()))
    if len(rows) != expected or len({d3._row_key(row) for row in rows}) != expected:
        raise RuntimeError(f"incomplete D3 sweep: {len(rows)} of {expected} unique records")
    if any(row.get("status") in ("worker failed", "not run (time)") for row in rows):
        raise RuntimeError("D3 sweep has failed workers or time-limited cells")
    manifest = json.loads(d3.REF_MANIFEST.read_text())
    confirmation = manifest["confirmation"]
    matched = matched_tables(rows, confirmation)
    hypotheses_result = hypotheses(rows, confirmation)
    plan = timing_plan(rows)
    (work_dir / "timing_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    determinism_result = determinism(rows, work_dir, strict)
    run_timing(rows, plan, work_dir)
    timing_result = timed_summary(plan, rows, work_dir)
    probes = [json.loads((work_dir / f"probe_{name}_{n}.json").read_text()) for name, n in d3.CASES]
    controls = [
        json.loads((work_dir / f"control_{name}_{n}_{rtol:.0e}.json").read_text())
        for name, n in d3.CASES
        if name != "P1"
        for rtol in (1e-4, 1e-6)
    ]
    payload = {
        "provenance": {
            **d3.check_environment(strict),
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "protocol": "D3 v3 frozen; v3.1 descriptive additions; v3.2 cost clarification; D2-only port",
        "validity_rule": "error_inf >= 100*est_err -> valid; 10*est_err <= error_inf < 100*est_err -> valid, reference-limited; error_inf < 10*est_err -> indeterminate",
        "settings": {
            "cases": [f"{name}/{n}" for name, n in d3.CASES],
            "rtol": d3.RTOLS,
            "atol_over_rtol": 1e-3,
            "targets": TARGETS,
            "H1_thresholds": {"support_Q": 1.15, "minimum_problem_count": 3, "loss_Q": 0.90},
            "NFE_per_attempt": d3.NFE,
            "transforms_per_field_per_attempt": d3.TRANSFORMS_PER_FIELD,
            "error_metric": "max fieldwise relative infinity error",
            "filters_and_imex_excluded": True,
        },
        "records": rows,
        "matched_targets": matched,
        "hypotheses": hypotheses_result,
        "reference_confirmation": confirmation,
        "estimator_power": probes,
        "rhs_time_controls": controls,
        "tolerance_proportionality": tolerance_proportionality(rows),
        "determinism": determinism_result,
        "timing": timing_result,
        "timing_plan": plan,
        "stage_a": status(rows),
        "raw_jsonl_sha256": hashlib.sha256((work_dir / "stage_a.jsonl").read_bytes()).hexdigest(),
    }
    output = d3.ROOT / "benchmarks/results/d3_work_precision.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if output.stat().st_size > 2_000_000:
        raise RuntimeError("D3 summary JSON exceeds 2 MB")
    _write_csv(rows, matched)
    (d3.ROOT / "benchmarks/results/D3_SUMMARY.md").write_text(
        render(json.loads(output.read_text()))
    )
    return payload


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "render":
        path = d3.ROOT / "benchmarks/results/d3_work_precision.json"
        print(render(json.loads(path.read_text())), end="")
    elif len(sys.argv) == 5 and sys.argv[1] == "timing-case":
        name, n, work_dir = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
        d3.check_environment()
        timing_case(
            read_rows(work_dir),
            json.loads((work_dir / "timing_plan.json").read_text()),
            work_dir,
            name,
            n,
        )
    else:
        raise SystemExit(
            "usage: d3_work_precision_analysis.py render | timing-case NAME N WORK_DIR"
        )


if __name__ == "__main__":
    main()
