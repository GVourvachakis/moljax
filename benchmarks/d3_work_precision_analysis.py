"""Checkpointed analysis and timing for the frozen D3 protocol v3.2."""

from __future__ import annotations

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
SELECTION = d3.SCRATCH / "results/e1_selection.json"
TIMING_PLAN = d3.SCRATCH / "results/timing_plan.json"
TIMING_RAW = d3.SCRATCH / "results/timing.jsonl"


def read_rows():
    return [json.loads(line) for line in d3.RAW.read_text().splitlines()]


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


def matched_tables(rows, epsilon):
    grouped = groups(rows)
    confirmed = {(r["problem"], r["N"]): r["epsilon_conf"] for r in epsilon}
    methods = ("S0", "S1a", "S1b", "S1c", "E0a", "E0b", "E1", "X", "sensitivity")
    tables = {}
    for name, n in d3.CASES:
        for method in methods:
            if not grouped[name, n, method]:
                continue
            cells = []
            for target in TARGETS:
                entry = {"target": target, "confirmed": target >= 10 * confirmed[name, n]}
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


def initial_hypotheses(rows):
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
            "eligible_problem_count": sum(
                cell["median_Q"] is not None for cell in by_problem.values()
            ),
        }

    h2 = {}
    for method in ("S1a", "S1b", "S1c"):
        per_case = {}
        for name, n in d3.CASES:
            values = []
            for target in TARGETS:
                q = q_value(grouped, (name, n), method, "S0", target, rtol_range=(1e-5, 1e-3))
                if q is not None:
                    values.append({"target": target, "Q": q})
            per_case[f"{name}/{n}"] = {
                "per_target": values,
                "median_Q": statistics.median(v["Q"] for v in values) if values else None,
            }
        focus = [per_case[key]["median_Q"] for key in ("P4/64", "P5/64", "P5/128")]
        helps = sum(value is not None and value >= 1.10 for value in focus)
        hurts = sum(value is not None and value <= 1 / 1.10 for value in focus)
        verdict = "helps" if helps >= 2 else "hurts" if hurts >= 2 else "neutral"
        h2[method] = {
            "per_case": per_case,
            "helps_count": helps,
            "hurts_count": hurts,
            "verdict": verdict,
            "median_focus_Q": statistics.median(value for value in focus if value is not None)
            if any(value is not None for value in focus)
            else None,
        }

    eligible_filters = [
        (name, row["median_focus_Q"])
        for name, row in h2.items()
        if row["verdict"] != "hurts" and row["median_focus_Q"] is not None
    ]
    best_filter = (
        sorted(
            eligible_filters, key=lambda item: (-item[1], {"S1a": 0, "S1b": 1, "S1c": 2}[item[0]])
        )[0][0]
        if eligible_filters
        else None
    )
    pair_scores = {}
    for name in ("E0a", "E0b"):
        values = [
            target["Q"] for row in h1[name]["by_problem"].values() for target in row["per_target"]
        ]
        pair_scores[name] = statistics.median(values) if values else None
    if all(score is None for score in pair_scores.values()):
        raise RuntimeError("No attainable embedded-pair targets for frozen E1 selection")
    best_pair = (
        "E0a"
        if pair_scores["E0b"] is None
        or pair_scores["E0a"] is not None
        and pair_scores["E0a"] >= pair_scores["E0b"]
        else "E0b"
    )
    selection = {
        "pair": best_pair,
        "method": d3.METHODS[best_pair],
        "controller": d3.CONTROLLERS[best_filter] if best_filter else None,
        "filter_config": best_filter,
        "pair_median_Q": pair_scores,
        "filter_verdicts": {name: row["verdict"] for name, row in h2.items()},
        "selection_basis": "median Q over each pair's eligible cases and targets; H2-best non-hurting filter",
    }
    return h1, h2, selection


def run_e1(selection):
    SELECTION.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    rows = read_rows()
    done = {(r["problem"], r["N"], r.get("rtol")) for r in rows if r["config"] == "E1"}
    cases = [case for case in d3.CASES if not (case[0] == "P1" and selection["pair"] == "E0b")]
    tasks = [(name, n, rtol) for name, n in cases for rtol in d3.RTOLS]
    grouped = groups(rows)

    def price(task):
        name, n, rtol = task
        source = next(
            (r for r in grouped[name, n, selection["pair"]] if r.get("rtol") == rtol), None
        )
        return 1.5 * source["wall_seconds_total"] if source else 900.0

    spent = sum(r.get("wall_seconds_total", 0.0) for r in rows)
    env = dict(os.environ, PYTHONPATH=str(d3.ROOT), JAX_PLATFORMS="cpu")
    for name, n, rtol in tasks:
        if (name, n, rtol) in done:
            continue
        remaining = [task for task in tasks if task not in done]
        projection = spent + sum(price(task) for task in remaining)
        if projection > 24 * 3600:
            gate = {
                "status": "HOLD",
                "phase": "E1",
                "spent_seconds": spent,
                "projected_total_hours": projection / 3600,
                "remaining_records": len(remaining),
            }
            (d3.SCRATCH / "results/stage_a_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
            return False
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(d3.ROOT / "benchmarks/d3_work_precision.py"),
                    "cell",
                    name,
                    str(n),
                    "E1",
                    str(rtol),
                ],
                cwd=d3.ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            row = (
                json.loads(proc.stdout.strip().splitlines()[-1])
                if proc.returncode == 0
                else {
                    "problem": name,
                    "N": n,
                    "config": "E1",
                    "rtol": rtol,
                    "status": "worker failed",
                    "reason": proc.stderr[-2000:],
                }
            )
        except subprocess.TimeoutExpired:
            row = {
                "problem": name,
                "N": n,
                "config": "E1",
                "rtol": rtol,
                "status": "not run (time)",
                "reason": "15-minute per-configuration guard",
            }
        row["wall_seconds_total"] = time.perf_counter() - started
        d3.append(d3.RAW, row)
        rows.append(row)
        grouped[name, n, "E1"].append(row)
        done.add((name, n, rtol))
        spent += row["wall_seconds_total"]
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "problem",
                        "N",
                        "config",
                        "rtol",
                        "status",
                        "success",
                        "accepted",
                        "rejected",
                        "wall_seconds_total",
                    )
                }
            ),
            flush=True,
        )
        if row.get("status") == "worker failed" and any(
            s in row.get("reason", "") for s in ("moljax imported from", "manifest mismatch")
        ):
            raise RuntimeError(row["reason"])
    return True


def determinism(rows):
    """Repeat one record for each method in a fresh Python process."""
    grouped = groups(rows)
    requested = [
        ("P1", 64, "F", 32),
        ("P5", 64, "F_g", 0),
        ("P1", 64, "S0", 1e-4),
        ("P1", 64, "S1a", 1e-4),
        ("P1", 64, "S1b", 1e-4),
        ("P1", 64, "S1c", 1e-4),
        ("P2", 128, "E0a", 1e-4),
        ("P2", 128, "E0b", 1e-4),
        ("P2", 128, "X", 1e-4),
    ]
    if grouped["P2", 128, "E1"]:
        requested.append(("P2", 128, "E1", 1e-4))
    env = dict(os.environ, PYTHONPATH=str(d3.ROOT), JAX_PLATFORMS="cpu")
    comparisons = []
    for name, n, config, value in requested:
        source = next(
            (
                r
                for r in grouped[name, n, config]
                if (r["steps"] if config == "F" else r["level"] if config == "F_g" else r["rtol"])
                == value
            ),
            None,
        )
        if source is None:
            raise RuntimeError(f"determinism source missing: {name}/{n}/{config}/{value}")
        proc = subprocess.run(
            [
                sys.executable,
                str(d3.ROOT / "benchmarks/d3_work_precision.py"),
                "cell",
                name,
                str(n),
                config,
                str(value),
            ],
            cwd=d3.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode:
            raise RuntimeError(f"determinism replay failed: {proc.stderr[-2000:]}")
        replay = json.loads(proc.stdout.strip().splitlines()[-1])
        match = (
            source.get("endpoint_sha256") == replay.get("endpoint_sha256")
            and source.get("accepted") == replay.get("accepted")
            and source.get("rejected") == replay.get("rejected")
        )
        comparisons.append(
            {
                "problem": name,
                "N": n,
                "config": config,
                "value": value,
                "counts_and_endpoint_match": match,
                "original_sha256": source.get("endpoint_sha256"),
                "replay_sha256": replay.get("endpoint_sha256"),
            }
        )
        if not match:
            (d3.SCRATCH / "results/determinism.json").write_text(
                json.dumps(comparisons, indent=2) + "\n"
            )
            raise RuntimeError(f"D3 determinism mismatch: {name}/{n}/{config}/{value}")
    (d3.SCRATCH / "results/determinism.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    return comparisons


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


def timing_plan(rows, selection):
    grouped = groups(rows)
    methods = ["F", "S0"]
    if selection["filter_config"]:
        methods.append(selection["filter_config"])
    methods.extend(("E0a", "E0b", "E1"))
    selected = []
    no_bracket = []
    for name, n in d3.CASES:
        for config in methods:
            if not grouped[name, n, config]:
                no_bracket.append(
                    {"problem": name, "N": n, "config": config, "reason": "method ineligible"}
                )
                continue
            pair = bracket(grouped[name, n, config])
            if pair is None:
                no_bracket.append(
                    {
                        "problem": name,
                        "N": n,
                        "config": config,
                        "reason": "e*=1e-4 outside valid Pareto error range",
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
                        "source_index": rows.index(row),
                    }
                )
    if len({entry["id"] for entry in selected}) != len(selected):
        raise RuntimeError("timing plan contains duplicate records")
    full_seconds = sum(50 * entry["steady_seconds"] for entry in selected)
    fallback = full_seconds > 8 * 3600
    for entry in selected:
        entry["runs_per_block"] = 4 if fallback and entry["steady_seconds"] > 5 else 10
        entry["priced_seconds"] = 5 * entry["runs_per_block"] * entry["steady_seconds"]
    priced = sum(entry["priced_seconds"] for entry in selected)
    plan = {
        "target": TIMING_TARGET,
        "selected": selected,
        "no_bracket": no_bracket,
        "full_5x10_seconds": full_seconds,
        "fallback_applied": fallback,
        "priced_seconds": priced,
        "priced_hours": priced / 3600,
        "passes_8h_gate": priced <= 8 * 3600,
        "rules": "5x10; if >8h, 5x4 only for records with steady run >5s; hold if still >8h",
    }
    TIMING_PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def timing_callable(row, selection):
    name, n, config = row["problem"], row["N"], row["config"]
    y0, ops, rhs, _, final = d3.setup(name, n)
    if config == "F":
        steps = row["steps"]

        def solve():
            return d3.etd_integrate(
                y0, (0.0, final), final / steps, ops, rhs, method="etdrk4", save_every=steps
            )[1][-1]

        return solve
    if config in ("S0", "S1a", "S1b", "S1c", "E0a", "E0b", "E1"):
        method = selection["method"] if config == "E1" else d3.METHODS.get(config, "etdrk4")
        controller = selection["controller"] if config == "E1" else d3.CONTROLLERS[config]
        pid = d3.PIDParams(atol=1e-3 * row["rtol"], rtol=row["rtol"], dt_max=final)

        def solve():
            return d3.etd_integrate_adaptive(
                y0,
                (0.0, final),
                1e-4 * final,
                ops,
                rhs,
                method=method,
                rhs_uses_time=method != "erk43zb",
                pid_params=pid,
                controller=controller,
                max_steps=row["max_steps"],
                save_every=row["save_every"],
            ).y_final

        return solve
    raise ValueError(f"not a Stage B timing configuration: {config}")


def timing_case(name, n):
    """Warm each graph, then run randomized interleaved blocks on one case."""
    d3.check_environment()
    plan = json.loads(TIMING_PLAN.read_text())
    selection = json.loads(SELECTION.read_text())
    rows = read_rows()
    selected = [entry for entry in plan["selected"] if entry["problem"] == name and entry["N"] == n]
    if not selected:
        return
    existing = set()
    if TIMING_RAW.exists():
        existing = {
            (row["id"], row["block"], row["repetition"])
            for row in (json.loads(line) for line in TIMING_RAW.read_text().splitlines())
        }
    expected = {
        (entry["id"], block, rep)
        for entry in selected
        for block in range(5)
        for rep in range(entry["runs_per_block"])
    }
    if expected <= existing:
        print(json.dumps({"timing_case": f"{name}/{n}", "checkpoint_complete": True}), flush=True)
        return
    functions = {}
    for entry in selected:
        source = rows[entry["source_index"]]
        assert record_id(source) == entry["id"]
        solve = timing_callable(source, selection)
        state = solve()
        d3.jax.tree.map(d3.jax.block_until_ready, state)
        if d3.endpoint_sha(state) != source["endpoint_sha256"]:
            raise RuntimeError(f"timing warmup endpoint mismatch: {entry['id']}")
        functions[entry["id"]] = solve
    for block in range(5):
        schedule = []
        for entry in selected:
            schedule.extend((entry["id"], rep) for rep in range(entry["runs_per_block"]))
        random.Random(20261116 + 1000 * n + 100 * block + ord(name[1])).shuffle(schedule)
        for identifier, rep in schedule:
            if (identifier, block, rep) in existing:
                continue
            start = time.perf_counter()
            state = functions[identifier]()
            d3.jax.tree.map(d3.jax.block_until_ready, state)
            seconds = time.perf_counter() - start
            ac = Path("/sys/class/power_supply/ACAD/online")
            entry = {
                "id": identifier,
                "problem": name,
                "N": n,
                "block": block,
                "repetition": rep,
                "seconds": seconds,
                "AC_online": ac.read_text().strip() if ac.exists() else None,
            }
            d3.append(TIMING_RAW, entry)
        print(json.dumps({"timing_case": f"{name}/{n}", "completed_block": block}), flush=True)


def run_timing(plan):
    if not plan["passes_8h_gate"]:
        return False
    env = dict(os.environ, PYTHONPATH=str(d3.ROOT), JAX_PLATFORMS="cpu")
    for name, n in d3.CASES:
        if not any(entry["problem"] == name and entry["N"] == n for entry in plan["selected"]):
            continue
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "timing-case", name, str(n)],
            cwd=d3.ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=8 * 3600,
        )
        if proc.returncode:
            raise RuntimeError(f"timing case {name}/{n} failed: {proc.stderr[-2000:]}")
        print(proc.stdout, flush=True)
    return True


def timed_summary(plan, rows):
    raw = [json.loads(line) for line in TIMING_RAW.read_text().splitlines()]
    by_id = defaultdict(list)
    for record in raw:
        by_id[record["id"]].append(record["seconds"])
    selected = {entry["id"]: entry for entry in plan["selected"]}
    summary = {}
    for identifier, entry in selected.items():
        times = np.asarray(by_id[identifier], dtype=float)
        expected = 5 * entry["runs_per_block"]
        if len(times) != expected:
            raise RuntimeError(f"incomplete timing record {identifier}: {len(times)} != {expected}")
        summary[identifier] = {
            "median_seconds": float(np.median(times)),
            "IQR_seconds": float(np.quantile(times, 0.75) - np.quantile(times, 0.25)),
            "n_runs": len(times),
            "AC_online_values": sorted(
                {r["AC_online"] for r in raw if r["id"] == identifier}, key=str
            ),
            "block_medians": [
                float(
                    np.median(
                        [r["seconds"] for r in raw if r["id"] == identifier and r["block"] == block]
                    )
                )
                for block in range(5)
            ],
        }
    grouped = groups(rows)
    interpolated = {}
    for name, n in d3.CASES:
        for config in {
            entry["config"]
            for entry in plan["selected"]
            if entry["problem"] == name and entry["N"] == n
        }:
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
    # Resample each record's runs, then interpolate the two medians at the target.
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
                values = []
                for identifier in interp["record_ids"]:
                    times = np.asarray(by_id[identifier])
                    draw = rng.choice(times, len(times), replace=True)
                    values.append(float(np.median(draw)))
                return math.exp(
                    math.log(values[0])
                    + interp["fraction"] * (math.log(values[1]) - math.log(values[0]))
                )

            ratios.append(sampled(baseline) / sampled(entry))
        ci[key] = {
            "Q_time_vs_S0": baseline["seconds"] / entry["seconds"],
            "bootstrap_95pct": [float(v) for v in np.quantile(ratios, [0.025, 0.975])],
        }
    return {
        "records": summary,
        "interpolated": interpolated,
        "ratios": ci,
        "bootstrap_seed": 20261116,
        "bootstrap_resamples": 2000,
    }


def _median(values):
    return statistics.median(values) if values else None


def _comparison(
    rows, candidate, baseline, metric="NFE", confirmed_only=False, epsilon=None, rtol_range=None
):
    grouped = groups(rows)
    levels = {(r["problem"], r["N"]): r["epsilon_conf"] for r in epsilon or []}
    out = {}
    for name, n in d3.CASES:
        points = []
        for target in TARGETS:
            if confirmed_only and target < 10 * levels[name, n]:
                continue
            q = q_value(grouped, (name, n), candidate, baseline, target, metric, rtol_range)
            if q is not None:
                points.append({"target": target, "Q": q})
        out[f"{name}/{n}"] = {"per_target": points, "median_Q": _median([p["Q"] for p in points])}
    return out


def _hypotheses(rows, h1, h2, selection, epsilon):
    lean = {
        method: _comparison(
            rows,
            method,
            "S0",
            "lean_NFE",
            rtol_range=(1e-5, 1e-3) if method.startswith("S1") else None,
        )
        for method in ("E0a", "E0b", "E1", "S1a", "S1b", "S1c")
    }
    confirmed = {
        method: _comparison(
            rows,
            method,
            "S0",
            "NFE",
            True,
            epsilon,
            (1e-5, 1e-3) if method.startswith("S1") else None,
        )
        for method in ("E0a", "E0b", "E1", "S1a", "S1b", "S1c")
    }
    grouped = groups(rows)
    h3 = {}
    for name, n in d3.CASES:
        for method in ("S0", "S1a", "S1b", "S1c", "E0a", "E0b", "E1"):
            if not grouped[name, n, method]:
                continue
            for baseline in ("F", "F_g") if name == "P5" else ("F",):
                values = []
                for target in TARGETS:
                    q = q_value(grouped, (name, n), method, baseline, target)
                    if q is not None:
                        values.append({"target": target, "R": q})
                h3[f"{name}/{n}/{method}/{baseline}"] = {
                    "per_target": values,
                    "median_R": _median([p["R"] for p in values]),
                }
    control_flags = {}
    for method, detail in h2.items():
        control_flags[method] = [
            key
            for key, cell in detail["per_case"].items()
            if key.split("/")[0] in ("P1", "P2", "P3")
            and cell["median_Q"] is not None
            and cell["median_Q"] < 1 / 1.05
        ]
    confirmed_verdicts = {}
    for method, cases in confirmed.items():
        if method in ("E0a", "E0b"):
            by_problem = {
                name: _median(
                    [
                        point["Q"]
                        for case, cell in cases.items()
                        if case.startswith(name + "/")
                        for point in cell["per_target"]
                    ]
                )
                for name in ("P1", "P2", "P3", "P4", "P5")
                if not (name == "P1" and method == "E0b")
            }
            confirmed_verdicts[method] = {
                "by_problem": by_problem,
                "supported": sum(v is not None and v >= 1.15 for v in by_problem.values()) >= 3
                and not any(v is not None and v < 0.90 for v in by_problem.values()),
            }
        elif method.startswith("S1"):
            focus = [cases[key]["median_Q"] for key in ("P4/64", "P5/64", "P5/128")]
            helps = sum(v is not None and v >= 1.10 for v in focus)
            hurts = sum(v is not None and v <= 1 / 1.10 for v in focus)
            confirmed_verdicts[method] = {
                "focus_median_Q": focus,
                "verdict": "helps" if helps >= 2 else "hurts" if hurts >= 2 else "neutral",
            }
    crossovers = []
    for method, detail in h1.items():
        for problem, cell in detail["by_problem"].items():
            values = [point["Q"] for point in cell["per_target"]]
            if values and min(values) < 1 < max(values):
                crossovers.append(
                    {
                        "configuration": method,
                        "problem": problem,
                        "min_Q": min(values),
                        "max_Q": max(values),
                    }
                )
    for method, detail in h2.items():
        for case, cell in detail["per_case"].items():
            values = [point["Q"] for point in cell["per_target"]]
            if values and min(values) < 1 < max(values):
                crossovers.append(
                    {
                        "configuration": method,
                        "case": case,
                        "min_Q": min(values),
                        "max_Q": max(values),
                    }
                )
    return {
        "H1": h1,
        "H2": h2,
        "H2_control_flags": control_flags,
        "H3": h3,
        "lean_NFE_Q": lean,
        "confirmed_target_Q": confirmed,
        "confirmed_target_verdicts": confirmed_verdicts,
        "crossovers": crossovers,
        "E1_selection": selection,
        "negative_result_clause": {
            "D2_not_proposed_upstream": not any(v["supported"] for v in h1.values()),
            "D1_at_most_documented_opt_in": all(v["verdict"] != "helps" for v in h2.values()),
        },
        "H3_structural_ceilings": {"step_doubling": 2 / 3, "ERK32ZB": 3 / 4, "ERK43ZB": 1},
    }


def _status(rows):
    counts = defaultdict(lambda: defaultdict(int))
    failures = []
    for row in rows:
        case = f"{row['problem']}/{row['N']}"
        counts[case][row["config"]] += 1
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
    exclusions = defaultdict(int)
    for row in rows:
        if row.get("validity") == "indeterminate":
            exclusions[f"{row['problem']}/{row['N']}/{row['config']}"] += 1
        elif row.get("error_inf_relative", 0) > 1e-2:
            exclusions[f"{row['problem']}/{row['N']}/{row['config']}"] += 1
    return {
        "counts": {case: dict(values) for case, values in counts.items()},
        "failures": failures,
        "excluded_counts": dict(exclusions),
    }


def _probe_data():
    probes = []
    controls = []
    for name, n in d3.CASES:
        probes.append(json.loads((d3.SCRATCH / f"results/probe_{name}_{n}.json").read_text()))
        if name != "P1":
            for rtol in (1e-4, 1e-6):
                controls.append(
                    json.loads(
                        (d3.SCRATCH / f"results/rhs_control_{name}_{n}_{rtol:.0e}.json").read_text()
                    )
                )
    return probes, controls


def _render(payload):
    lines = [
        "# D3 adaptive ETD work-precision study",
        "",
        "The frozen v3 protocol was followed by descriptive v3.1 additions and the v3.2 cost-plan revision before this sweep. The first cost gate exposed equal-tolerance counts at rtol 1e-8 without errors. No protocol threshold, target, tolerance, configuration or primary metric changed in response.",
        "",
        "All measurements used CPU and JAX 0.11.2. NFE is the primary work metric; timings are secondary.",
        "",
        "## Provenance and gates",
        "",
        f"- Branch: `{payload['provenance']['execution_head']}`; Python {payload['provenance']['python']}; JAX {payload['provenance']['jax']}; jaxlib {payload['provenance']['jaxlib']}.",
        "Source: `benchmarks/results/d3_work_precision.json`, `/provenance`.",
        "",
        f"- Stage A projection: {payload['gates']['stage_a_projection']['runner_initial_projection_hours']:.3g} h against 24 h; Stage B price: {payload['gates']['stage_b']['priced_hours']:.3g} h against 8 h. Fallback: {payload['gates']['stage_b']['fallback_applied']}.",
        "Source: `benchmarks/results/d3_work_precision.json`, `/gates`.",
        "",
        f"- Earlier combined cost estimate: {payload['gates']['first_combined_projection_hours']:.3g} h; revised Stage A planning upper bound: {payload['gates']['stage_a_projection']['stage_a_upper_hours']:.3g} h. These were planning estimates, not D3 results.",
        "Source: `benchmarks/results/d3_work_precision.json`, `/gates/first_combined_projection_hours` and `/gates/stage_a_projection/stage_a_upper_hours`.",
        "",
        "## Reference confirmation",
        "",
        "| Case | epsilon_conf | Basis | Confirmed targets |",
        "| --- | ---: | --- | --- |",
    ]
    for row in payload["reference_confirmation"]:
        targets = [t for t in TARGETS if t >= 10 * row["epsilon_conf"]]
        lines.append(
            f"| {row['problem']}/{row['N']} | {row['epsilon_conf']:.4g} | {row['method']} | {', '.join(f'{t:.0e}' for t in targets) or 'none'} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/reference_confirmation` (rows in case order).",
        "",
        f"D0b validity rule, verbatim: `{payload['validity_rule']}`. This rule uses absolute error. Matched accuracy uses the maximum per-field relative infinity error.",
        "Source: `benchmarks/results/d3_work_precision.json`, `/validity_rule` and `/settings/error_metric`.",
        "",
        "## Stage A records",
        "",
        "| Case | Record counts by configuration | Excluded |",
        "| --- | --- | ---: |",
    ]
    for case, counts in payload["stage_a"]["counts"].items():
        excluded = sum(
            value
            for key, value in payload["stage_a"]["excluded_counts"].items()
            if key.startswith(case + "/")
        )
        lines.append(
            f"| {case} | "
            + ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
            + f" | {excluded} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/stage_a/counts` and `/stage_a/excluded_counts`.",
        "",
        "| Failed case | Configuration | Tolerance/step | Status |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["stage_a"]["failures"]:
        value = (
            row["rtol"]
            if row["rtol"] is not None
            else row["steps"]
            if row["steps"] is not None
            else row["level"]
        )
        lines.append(
            f"| {row['problem']}/{row['N']} | {row['config']} | {value} | {row['status']} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/stage_a/failures`.",
        "",
        "## Matched accuracy and hypotheses",
        "",
        "Q(A versus B) is B's interpolated NFE divided by A's at the same error; Q above one favours A. The tables below contain only targets in the recorded Pareto range.",
        "",
        "| Pair | Problem | Median Q against S0 | H1 |",
        "| --- | --- | ---: | --- |",
    ]
    for method, detail in payload["hypotheses"]["H1"].items():
        for problem, cell in detail["by_problem"].items():
            value = cell["median_Q"]
            lines.append(
                f"| {method} | {problem} | {value:.4g} | {detail['supported']} |"
                if value is not None
                else f"| {method} | {problem} | unavailable | {detail['supported']} |"
            )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H1`.",
        "",
        "| Filter | P4/64 Q | P5/64 Q | P5/128 Q | H2 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for method, detail in payload["hypotheses"]["H2"].items():
        values = [detail["per_case"][case]["median_Q"] for case in ("P4/64", "P5/64", "P5/128")]
        lines.append(
            f"| {method} | "
            + " | ".join("unavailable" if v is None else f"{v:.4g}" for v in values)
            + f" | {detail['verdict']} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H2`.",
        "",
    ]
    if all(
        not detail["per_case"][case]["per_target"]
        for detail in payload["hypotheses"]["H2"].values()
        for case in ("P4/64", "P5/64", "P5/128")
    ):
        lines += [
            "The H2 focus cases have no matched target in the prescribed tolerance window. The frozen rule returns neutral here; that is an unassessed comparison, not evidence of equal cost.",
            "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H2`.",
            "",
        ]
    lines += [
        "### Per-target work ratios",
        "",
        "| Case | Configuration | Target | Q versus S0 | Lean-NFE Q | Confirmed target |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for key, cells in payload["matched_targets"].items():
        case, config = key.rsplit("/", 1)
        for cell in cells:
            primary = cell["Q_vs_S0_NFE"]
            lean = cell["Q_vs_S0_lean_NFE"]
            if config == "S0" or primary is None:
                continue
            lines.append(
                f"| {case} | {config} | {cell['target']:.0e} | {primary:.4g} | "
                f"{lean:.4g} | {cell['confirmed']} |"
                if lean is not None
                else f"| {case} | {config} | {cell['target']:.0e} | {primary:.4g} | n/a | {cell['confirmed']} |"
            )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/matched_targets`.",
        "",
        "| Case | Configuration | Attainable targets |",
        "| --- | --- | --- |",
    ]
    for key, cells in payload["matched_targets"].items():
        case, config = key.rsplit("/", 1)
        attainable = [f"{cell['target']:.0e}" for cell in cells if cell["NFE"] is not None]
        lines.append(f"| {case} | {config} | {', '.join(attainable) or 'none'} |")
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/matched_targets`.",
        "",
        "### H3: fixed-step anchors",
        "",
        "| Case | Configuration | Anchor | Target | R |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for key, detail in payload["hypotheses"]["H3"].items():
        case, config, baseline = key.rsplit("/", 2)
        for cell in detail["per_target"]:
            lines.append(
                f"| {case} | {config} | {baseline} | {cell['target']:.0e} | {cell['R']:.4g} |"
            )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H3`.",
        "",
        "H3 structural ceilings against each method's own fixed-step use: "
        + ", ".join(
            f"{key} {value:.4g}"
            for key, value in payload["hypotheses"]["H3_structural_ceilings"].items()
        )
        + ".",
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/H3_structural_ceilings`.",
        "",
        "### Robustness of the primary comparisons",
        "",
        "| Configuration | Case | Median lean-NFE Q | Median confirmed-target Q |",
        "| --- | --- | ---: | ---: |",
    ]
    for config, cases in payload["hypotheses"]["lean_NFE_Q"].items():
        for case, detail in cases.items():
            lean = detail["median_Q"]
            confirmed = payload["hypotheses"]["confirmed_target_Q"][config][case]["median_Q"]
            lines.append(
                f"| {config} | {case} | {lean:.4g} | {confirmed:.4g} |"
                if lean is not None and confirmed is not None
                else f"| {config} | {case} | {lean if lean is not None else 'n/a'} | "
                f"{confirmed if confirmed is not None else 'n/a'} |"
            )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/lean_NFE_Q` and `/hypotheses/confirmed_target_Q`.",
        "",
        "| Configuration | Confirmed-target robustness verdict |",
        "| --- | --- |",
    ]
    for config, detail in payload["hypotheses"]["confirmed_target_verdicts"].items():
        lines.append(f"| {config} | {detail.get('verdict', detail.get('supported'))} |")
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/confirmed_target_verdicts`.",
        "",
        "| Crossover configuration | Problem/case | Lowest Q | Highest Q |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in payload["hypotheses"]["crossovers"]:
        lines.append(
            f"| {item['configuration']} | {item.get('case', item.get('problem'))} | {item['min_Q']:.4g} | {item['max_Q']:.4g} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/crossovers`.",
        "",
        f"E1 selected `{payload['hypotheses']['E1_selection']['pair']}` with filter `{payload['hypotheses']['E1_selection']['filter_config']}`. D2 not proposed upstream: {payload['hypotheses']['negative_result_clause']['D2_not_proposed_upstream']}; D1 at most documented opt-in: {payload['hypotheses']['negative_result_clause']['D1_at_most_documented_opt_in']}.",
        "Source: `benchmarks/results/d3_work_precision.json`, `/hypotheses/E1_selection` and `/hypotheses/negative_result_clause`.",
        "",
        "## Timing",
        "",
        "| Case/config | Median seconds | IQR seconds | Runs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, row in payload["timing"]["records"].items():
        lines.append(
            f"| {key} | {row['median_seconds']:.4g} | {row['IQR_seconds']:.3g} | {row['n_runs']} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/timing/records`.",
        "",
        "| Unbracketed case | Configuration | Reason |",
        "| --- | --- | --- |",
    ]
    for item in payload["gates"]["stage_b"]["no_bracket"]:
        lines.append(f"| {item['problem']}/{item['N']} | {item['config']} | {item['reason']} |")
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/gates/stage_b/no_bracket`.",
        "",
        "| Case/config | Interpolated seconds at target | Time Q versus S0 | Bootstrap 95% CI |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, detail in payload["timing"]["interpolated"].items():
        ratio = payload["timing"]["ratios"].get(key)
        lines.append(
            f"| {key} | {detail['seconds']:.4g} | "
            + (
                f"{ratio['Q_time_vs_S0']:.4g} | [{ratio['bootstrap_95pct'][0]:.4g}, {ratio['bootstrap_95pct'][1]:.4g}] |"
                if ratio
                else "n/a | n/a |"
            )
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/timing/interpolated` and `/timing/ratios`.",
        "",
        "## Estimator-power probes and time dependence",
        "",
        "| Case | Method | State fraction | Six slopes | Median k>=2 | Nominal |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for probe in payload["estimator_power"]:
        for entry in probe["entries"]:
            lines.append(
                f"| {probe['problem']}/{probe['N']} | {entry['method']} | {entry['target_fraction']:.2g} | "
                + ", ".join(f"{s:.3g}" for s in entry["slopes"])
                + f" | {entry['median_slope_k_ge_2']:.3g} | {entry['nominal_power']} |"
            )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/estimator_power`.",
        "",
        "| Case | rtol | Attempts equal | Endpoints equal |",
        "| --- | ---: | --- | --- |",
    ]
    for row in payload["rhs_time_controls"]:
        lines.append(
            f"| {row['problem']}/{row['N']} | {row['rtol']:.0e} | "
            f"{row['attempts_equal']} | {row['endpoint_equal']} |"
        )
    lines += [
        "Source: `benchmarks/results/d3_work_precision.json`, `/rhs_time_controls`.",
        "",
        "## Scope",
        "",
        "The JSON carries per-target Q and R against F/F_g, lean-NFE robustness values, confirmed-target values, attainable targets and failure rows. No extrapolation was used.",
        "Source: `benchmarks/results/d3_work_precision.json`, `/matched_targets`, `/hypotheses`, and `/stage_a`.",
        "",
    ]
    return "\n".join(lines)


def complete_study():
    d3.check_environment()
    rows = read_rows()
    grouped = groups(rows)
    original = json.loads((d3.EVIDENCE / "results/d0b.json").read_text())
    fixed_counts = {(r["problem"], r["N"]): len(r["cells"]) for r in original["B2"]["fixed_ladder"]}
    for name, n in d3.CASES:
        if len(grouped[name, n, "F"]) != fixed_counts[name, n]:
            raise RuntimeError(f"Stage A fixed ladder incomplete: {name}/{n}")
        if name == "P5" and len(grouped[name, n, "F_g"]) != 5:
            raise RuntimeError(f"Stage A graded ladder incomplete: {name}/{n}")
        for config in d3.CONFIGS:
            if name == "P1" and config in ("E0b", "X"):
                continue
            if len(grouped[name, n, config]) != len(d3.RTOLS):
                raise RuntimeError(f"Stage A incomplete: {name}/{n}/{config}")
    epsilon = json.loads((d3.SCRATCH / "results/reference_confirmation.json").read_text())
    h1, h2, selection = initial_hypotheses(rows)
    if not run_e1(selection):
        print("HOLD: Stage A E1 projection exceeded 24 h", flush=True)
        return
    rows = read_rows()
    checked = determinism(rows)
    plan = timing_plan(rows, selection)
    if not plan["passes_8h_gate"]:
        print(f"HOLD: Stage B priced at {plan['priced_hours']:.3f} h after fallback", flush=True)
        return
    run_timing(plan)
    probes, controls = _probe_data()
    projection = json.loads((d3.SCRATCH / "results/stage_a_projection.json").read_text())
    for line in (d3.SCRATCH / "logs/stage_a.log").read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "stage_a_initial_projection_hours" in item:
            projection["runner_initial_projection_hours"] = item["stage_a_initial_projection_hours"]
            break
    else:
        raise RuntimeError("Stage A run-time projection missing from log")
    payload = {
        "provenance": d3.check_environment(),
        "protocol_version": "v3.2",
        "validity_rule": json.loads((d3.SCRATCH / "results/pre_run_v3_1.json").read_text())[
            "D0b_validity_rule"
        ],
        "settings": {
            "cases": [f"{name}/{n}" for name, n in d3.CASES],
            "rtol": list(d3.RTOLS),
            "atol_over_rtol": 1e-3,
            "dt0_over_T": 1e-4,
            "dt_max": "T",
            "targets": list(TARGETS),
            "matched_accuracy_error_cap": 1e-2,
            "validity_factors_absolute": {"valid": 100, "reference_limited": 10},
            "H1_thresholds": {"support_Q": 1.15, "minimum_problem_count": 3, "loss_Q": 0.90},
            "H2_thresholds": {
                "help_Q": 1.10,
                "hurt_Q": 1 / 1.10,
                "control_flag_Q": 1 / 1.05,
                "minimum_focus_count": 2,
            },
            "cost_gates_hours": {"stage_a": 24, "stage_b": 8},
            "error_metric": "max over fields of infinity error / infinity reference norm",
            "primary_metric": "nominal NFE",
            "memory_guard_bytes": d3.MEMORY_LIMIT,
        },
        "constants": {
            "NFE_per_attempt": {
                "etd1": 3,
                "etdrk4": 12,
                "erk32zb": 4,
                "erk43zb": 5,
                "imex_strang_X": 6,
            },
            "transforms_per_attempt_per_field": {
                "etdrk4": [15, 12],
                "erk32zb": [5, 4],
                "erk43zb": [6, 5],
            },
            "controller_estimate_orders": {"S0": 5, "E0a": 3, "E0b": 4},
            "H3_structural_ceilings": {"step_doubling": 2 / 3, "erk32zb": 3 / 4, "erk43zb": 1},
        },
        "reference_confirmation": epsilon,
        "stage_a": _status(rows),
        "matched_targets": matched_tables(rows, epsilon),
        "hypotheses": _hypotheses(rows, h1, h2, selection, epsilon),
        "gates": {
            "first_combined_projection_hours": json.loads(
                (d3.SCRATCH / "results/cost_gate_projection.json").read_text()
            )["planning_total_hours"],
            "stage_a_projection": projection,
            "stage_b": plan,
        },
        "timing": timed_summary(plan, rows),
        "determinism": checked,
        "estimator_power": probes,
        "rhs_time_controls": controls,
        "records": rows,
        "raw_jsonl_sha256": d3.hashlib.sha256(d3.RAW.read_bytes()).hexdigest(),
        "analysis_script_sha256": d3.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target = d3.ROOT / "benchmarks/results/d3_work_precision.json"
    target.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if target.stat().st_size > 2_000_000:
        raise RuntimeError(f"D3 summary exceeds 2 MB: {target.stat().st_size}")
    (d3.ROOT / "D3_SUMMARY.md").write_text(_render(json.loads(target.read_text())))
    print(f"D3 complete: {len(rows)} records, {target.stat().st_size} JSON bytes", flush=True)
    manifest = []
    for folder in ("harness", "results", "logs"):
        for path in sorted((d3.SCRATCH / folder).rglob("*")):
            if (
                not path.is_file()
                or path.name == "SHA256SUMS"
                or path.suffix == ".pid"
                or "__pycache__" in path.parts
            ):
                continue
            manifest.append(
                f"{d3.hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(d3.SCRATCH)}"
            )
    (d3.SCRATCH / "results/SHA256SUMS").write_text("\n".join(manifest) + "\n")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "timing-case":
        timing_case(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1:] == ["complete"]:
        complete_study()
    else:
        raise SystemExit("usage: d3_work_precision_analysis.py complete | timing-case PROBLEM N")


if __name__ == "__main__":
    main()
