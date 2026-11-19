"""Render D3 and D3b Markdown from committed result files only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from benchmarks import d3_work_precision as d3

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT = RESULTS / "d3b_graded_decomposition.json"
REPORT = RESULTS / "D3B_SUMMARY.md"
HEADING = "## Graded two-factor view on P5 (exploratory, post hoc)"
NEXT_HEADING = "## Missing matched-target cells"


def _fmt(value, pointer, digits=4):
    return (
        f"{value:.{digits}g} (`{pointer}`)" if value is not None else f"unattainable (`{pointer}`)"
    )


def _key(value):
    """Escape one mapping key for an RFC 6901 JSON pointer."""
    return str(value).replace("~", "~0").replace("/", "~1")


def _case_order(key):
    case = key.split("/", 2)
    return d3.CASES.index((case[0], int(case[1]))), key


def _record_order(key):
    name, n, *_ = key.split("_")
    return d3.CASES.index((name, int(n))), key


def _target(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def render(payload):
    reference_manifest = json.loads((RESULTS / "d3_references.json").read_text())
    if payload["reference_confirmation"] != reference_manifest["confirmation"]:
        raise RuntimeError("reference confirmation differs from the committed manifest")
    study_timing = json.loads((RESULTS / "d3_timing_study_session.json").read_text())["timing"]
    reproduction = json.loads((RESULTS / "d3_reproduction.json").read_text())
    lines = [
        "# D3: adaptive exponential work precision",
        "",
        "## Setup and protocol",
        "",
        "The H1 threshold and matched accuracy targets were fixed before comparison. "
        "The reference arrays and their checksums are in `d3_references.npz` and "
        "`d3_references.json`. CPU timings are secondary evidence.",
        "",
        "| Problem | Model | Fields | Domain length | Final time | Diffusivities | Grids | RHS reads time | Regime |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for name, title, length, final, diffusivities, reads_time, regime in (
        (name, *settings) for name, settings in d3.PROBLEM_SETTINGS.items()
    ):
        grids = ", ".join(str(n) for case, n in d3.CASES if case == name)
        diffusion = ", ".join(f"{field}={value:g}" for field, value in diffusivities.items())
        lines.append(
            f"| {name} | {title} | {', '.join(diffusivities)} | {length:g} | {final:g} | "
            f"{diffusion} | {grids} | {reads_time} | {regime or '—'} |"
        )
    lines += [
        "",
        "Source: `benchmarks/d3_work_precision.py`, `PROBLEM_SETTINGS`, `CASES`, `setup()`; "
        "periodic FD `DiffusionOperator` on every case.",
        "",
        "| Configuration | Meaning | NFE per attempt |",
        "| --- | --- | ---: |",
    ]
    configurations = (
        ("S0", "ETDRK4 step doubling with the default controller"),
        ("sensitivity", "S0 with atol = rtol"),
        ("E0a", "ERK32ZB embedded pair"),
        ("E0b", "ERK43ZB embedded pair"),
    )
    for key, meaning in configurations:
        lines.append(
            f"| {key} | {meaning} | {_fmt(payload['settings']['NFE_per_attempt'][key], '/settings/NFE_per_attempt/' + key)} |"
        )
    lines += [
        "| F | Uniform fixed ETDRK4 | 4 per fixed step (`benchmarks/d3_work_precision.py:fixed_cell`) |",
        "| F_g | Graded fixed ETDRK4 on P5 | 4 per fixed step (`benchmarks/d3_work_precision.py:fixed_cell`) |",
        "",
        "Source: `d3_work_precision.json`, `/settings/NFE_per_attempt`; `benchmarks/d3_work_precision.py`, `fixed_cell`.",
        "",
        f"The error metric is {payload['settings']['error_metric']} (`/settings/error_metric`). "
        f"The validity rule is {payload['validity_rule']} (`/validity_rule`): it compares absolute "
        "error with the reference's `est_err` and selects records for the Pareto fronts. "
        "Q is S0 NFE divided by pair NFE at matched error; R uses F or F_g NFE as the numerator. "
        "Both use log–log interpolation on valid Pareto fronts without extrapolation. "
        "Lean NFE counts one reaction evaluation per step position: "
        "S0 uses `10*attempts + accepted`, E0a `3*attempts + 1`, and E0b "
        "`4*attempts + accepted` (`benchmarks/d3_work_precision.py:adaptive_cell`).",
        "",
        "A target is confirmed when it is at least ten times its reference-confirmation level "
        "(`reference_confirmation/*/epsilon_conf`). Confirmation labels come from the "
        "[reference audit on the study branch](https://github.com/GVourvachakis/moljax/tree/"
        "f3a05c93a07a2cf3684660f94f410b989bd119e6). H1 uses every attainable valid target "
        "regardless of confirmation (`benchmarks/d3_work_precision_analysis.py:hypotheses`).",
        "",
        f"H1 requires Q ≥ {_fmt(payload['settings']['H1_thresholds']['support_Q'], '/settings/H1_thresholds/support_Q')} "
        f"on at least {_fmt(payload['settings']['H1_thresholds']['minimum_problem_count'], '/settings/H1_thresholds/minimum_problem_count')} "
        f"eligible problems, with no problem below {_fmt(payload['settings']['H1_thresholds']['loss_Q'], '/settings/H1_thresholds/loss_Q')}.",
        "",
        f"Frozen targets: {', '.join(_target(target) for target in payload['settings']['targets'])} "
        "(`/settings/targets`).",
        "",
        "| Case | Reference confirmation level | Method |",
        "| --- | ---: | --- |",
    ]
    for name, n in d3.CASES:
        case = f"{name}_{n}"
        detail = payload["reference_confirmation"][case]
        lines.append(
            f"| {case} | {_fmt(detail['epsilon_conf'], '/reference_confirmation/' + case + '/epsilon_conf')} | {detail['method']} |"
        )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/settings`, `/reference_confirmation`, `/validity_rule`.",
        "",
        f"All {len(payload['records'])} records were assembled (`/records`); every adaptive run succeeded. "
        f"The {len(payload['stage_a']['failures'])} unsuccessful records are coarse fixed-step "
        "rungs that became non-finite (`/stage_a/failures`). Exclusions by case and configuration:",
        "",
        "| Case / configuration | Excluded records |",
        "| --- | ---: |",
    ]
    for key, count in sorted(
        payload["stage_a"]["excluded_counts"].items(), key=lambda item: _case_order(item[0])
    ):
        lines.append(f"| {key} | {count} |")
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/stage_a/excluded_counts`.",
        "",
        "## H1: embedded pairs against step doubling",
        "",
        f"| Pair | Problem | Median Q | ≥ {payload['settings']['H1_thresholds']['support_Q']:.3g} | "
        f"< {payload['settings']['H1_thresholds']['loss_Q']:.3g} |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for method, result in payload["hypotheses"]["H1"].items():
        for problem, cell in result["by_problem"].items():
            ptr = f"/hypotheses/H1/{method}/by_problem/{problem}/median_Q"
            lines.append(
                f"| {method} | {problem} | {_fmt(cell['median_Q'], ptr)} | "
                f"{cell['median_Q'] is not None and cell['median_Q'] >= payload['settings']['H1_thresholds']['support_Q']} | "
                f"{cell['median_Q'] is not None and cell['median_Q'] < payload['settings']['H1_thresholds']['loss_Q']} |"
            )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/hypotheses/H1/*/by_problem/*/median_Q`.",
        "",
    ]
    for method, result in payload["hypotheses"]["H1"].items():
        base = f"/hypotheses/H1/{method}"
        lines.append(
            f"{method}: {_fmt(result['positive_problem_count'], base + '/positive_problem_count')} of "
            f"{_fmt(result['eligible_problem_count'], base + '/eligible_problem_count')} eligible problems "
            f"meet the positive threshold; any below the loss threshold: "
            f"{result['any_problem_below_0p90']} (`{base}/any_problem_below_0p90`); "
            f"supported: {result['supported']} (`{base}/supported`)."
        )
    lines += [
        "ERK43ZB has no P1 row because its time-dependent RHS is incompatible with the pair's "
        "one-sixth stage under the tick-clock rounding guarantee.",
        "",
        "The primary Q uses nominal NFE. Lean NFE and CPU-time ratios are reported as secondary checks.",
        "",
        "| JSON key | Target | Q | Lean Q | Confirmed |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, cells in sorted(
        payload["matched_targets"].items(), key=lambda item: _case_order(item[0])
    ):
        if not key.endswith(("/E0a", "/E0b")):
            continue
        for cell in cells:
            lines.append(
                f"| {key} | {_target(cell['target'])} | {cell['Q_vs_S0_NFE']:.4g} | "
                if cell["Q_vs_S0_NFE"] is not None
                else f"| {key} | {_target(cell['target'])} | unattainable | "
            )
            lines[-1] += (
                f"{cell['Q_vs_S0_lean_NFE']:.4g}"
                if cell["Q_vs_S0_lean_NFE"] is not None
                else "unattainable"
            ) + f" | {cell['confirmed']} |"
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/matched_targets/*/*/Q_vs_S0_NFE`, "
        "`/matched_targets/*/*/Q_vs_S0_lean_NFE`, `/matched_targets/*/*/confirmed`.",
        "",
        "## H3: fixed-step anchors (descriptive)",
        "",
        "| Case / method / baseline | Median R |",
        "| --- | ---: |",
    ]
    for key, cell in sorted(
        payload["hypotheses"]["H3"].items(), key=lambda item: _case_order(item[0])
    ):
        lines.append(
            f"| {key} | {cell['median_R']:.4g} |"
            if cell["median_R"] is not None
            else f"| {key} | unattainable |"
        )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/hypotheses/H3/*/median_R`.",
        "",
        "## Estimator power and tolerance proportionality",
        "",
        "| Case | Method | Time fraction | Median slope | Nominal power |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for probe in payload["estimator_power"]:
        for entry in probe["entries"]:
            lines.append(
                f"| {probe['problem']}/{probe['N']} | {entry['method']} | "
                f"{entry['target_fraction']:.3g} | {entry['median_slope_k_ge_2']:.4g} | "
                f"{entry['nominal_power']} |"
            )
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/estimator_power/*/entries/*/median_slope_k_ge_2`.",
        "",
        "| Case / method | Tolerance slope | Valid records |",
        "| --- | ---: | ---: |",
    ]
    for key, cell in sorted(
        payload["tolerance_proportionality"].items(), key=lambda item: _case_order(item[0])
    ):
        lines.append(f"| {key} | {cell['slope']:.4g} | {cell['n']} |")
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/tolerance_proportionality/*/slope`, "
        "`/tolerance_proportionality/*/n`.",
        "",
        "## CPU timing ratios",
        "",
        "| Case / method | Time ratio against S0 | Bootstrap interval |",
        "| --- | ---: | ---: |",
    ]
    for key, cell in sorted(
        payload["timing"]["ratios"].items(), key=lambda item: _case_order(item[0])
    ):
        lo, hi = cell["bootstrap_95pct"]
        lines.append(f"| {key} | {cell['Q_time_vs_S0']:.4g} | {lo:.4g} to {hi:.4g} |")
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/timing/ratios/*/Q_time_vs_S0`, "
        "`/timing/ratios/*/bootstrap_95pct`.",
        "",
        "## Timing protocol and between-session check",
        "",
    ]
    with (RESULTS / "d3_timing_runs.csv").open(newline="") as stream:
        timing_runs = list(csv.DictReader(stream))
    blocks = len({row["block"] for row in timing_runs})
    repetitions = max(int(row["repetition"]) for row in timing_runs) + 1
    lines += [
        "Each timed record was warmed up and checked against its endpoint hash "
        "(`benchmarks/d3_work_precision_analysis.py:timing_case`). "
        f"The runs used {blocks} shuffled blocks and up to {repetitions} runs per block "
        "(`d3_timing_runs.csv`), then the median time was interpolated in log–log "
        f"space at the matched {_fmt(payload['timing_plan']['target'], '/timing_plan/target')} "
        "target between bracketing Pareto records "
        "(`benchmarks/d3_work_precision_analysis.py:timed_summary`). "
        f"Intervals use {_fmt(payload['timing']['bootstrap_resamples'], '/timing/bootstrap_resamples')} "
        "bootstrap resamples of runs.",
        "",
        "| Case / configuration | No-bracket reason |",
        "| --- | --- |",
    ]
    for row in payload["timing_plan"]["no_bracket"]:
        lines.append(f"| {row['case']}/{row['config']} | {row['reason']} |")
    lines += [
        "",
        "Source: `d3_work_precision.json`, `/timing_plan/no_bracket/*/reason`.",
        "",
        "| Case / pair | Study Q [interval] | This run Q [interval] | Overlap |",
        "| --- | --- | --- | --- |",
    ]
    for key, current in sorted(
        payload["timing"]["ratios"].items(), key=lambda item: _case_order(item[0])
    ):
        old = study_timing["ratios"][key]
        old_path = "/timing/ratios/" + _key(key)
        new_path = old_path
        old_lo, old_hi = old["bootstrap_95pct"]
        new_lo, new_hi = current["bootstrap_95pct"]
        overlap = max(old_lo, new_lo) <= min(old_hi, new_hi)
        lines.append(
            f"| {key} | {_fmt(old['Q_time_vs_S0'], old_path + '/Q_time_vs_S0')} "
            f"[{_fmt(old_lo, old_path + '/bootstrap_95pct/0')}, "
            f"{_fmt(old_hi, old_path + '/bootstrap_95pct/1')}] | "
            f"{_fmt(current['Q_time_vs_S0'], new_path + '/Q_time_vs_S0')} "
            f"[{_fmt(new_lo, new_path + '/bootstrap_95pct/0')}, "
            f"{_fmt(new_hi, new_path + '/bootstrap_95pct/1')}] | {overlap} |"
        )
    lines += [
        "",
        "Source: study `d3_timing_study_session.json` and this run "
        "`d3_work_precision.json`, `/timing/ratios/*`.",
        "",
        "| Timed record ID | Study IQR/median | This run IQR/median |",
        "| --- | ---: | ---: |",
    ]
    for key, current in sorted(
        payload["timing"]["records"].items(), key=lambda item: _record_order(item[0])
    ):
        old = study_timing["records"][key]
        lines.append(
            f"| {key} | {old['IQR_seconds'] / old['median_seconds']:.3g} | "
            f"{current['IQR_seconds'] / current['median_seconds']:.3g} |"
        )
    lines += [
        "",
        "Source: study `d3_timing_study_session.json` and this run "
        "`d3_work_precision.json`, `/timing/records/*/IQR_seconds`, "
        "`/timing/records/*/median_seconds`.",
        "",
        "Within-session intervals do not cover between-session differences on this laptop; "
        "CPU-time ratios are not evidence for H1. The NFE records are bitwise reproducible.",
        "",
        "## Reproduction",
        "",
        f"The {_fmt(reproduction['d3']['records_identical'], '/d3/records_identical')} "
        f"of {_fmt(reproduction['d3']['records_compared'], '/d3/records_compared')} records "
        "reproduce the [study](https://github.com/GVourvachakis/moljax/tree/"
        "f3a05c93a07a2cf3684660f94f410b989bd119e6) bitwise on the same "
        "laptop CPU and JAX 0.11.2. Other hosts can differ at rounding level, "
        "because the references are host specific.",
        "",
        "Source: `d3_reproduction.json`, `/d3`; `d3_references.json`, `/host_specific`.",
        "",
        "## Limits",
        "",
        "ERK43ZB's 1/6 stage is outside the tick clock's one-rounding guarantee, so it is "
        "restricted to autonomous RHS. Measurements used a laptop CPU. Filter presets and the "
        "adaptive IMEX comparator remain on [experiment/etd-embedded-pairs](https://github.com/GVourvachakis/moljax/tree/f3a05c93a07a2cf3684660f94f410b989bd119e6); they are not part of this branch.",
        "",
        "Source: `d3_work_precision.json`, `/settings`.",
        "",
    ]
    return "\n".join(lines)


def render_d3b_base(payload):
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
        "The ladders were measured by `d3b_fixed_pairs.py` at its recorded "
        f"script SHA-256 `{payload['provenance']['script_sha256']}`; that blob is "
        "committed in `b494c0c`, before rendering moved. D3b read the D3 JSON "
        f"as committed in `deb96e7` (SHA-256 `{payload['provenance']['d3_json_sha256']}`).",
        "Source: `d3b_fixed_pairs.json`, `/provenance/script_sha256`, `/provenance/d3_json_sha256`.",
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
        "",
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
        "",
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
    lines += ["", "Source: `d3b_fixed_pairs.json`, `/missing` (every row).", ""]
    return "\n".join(lines)


def render_section(payload):
    rows = payload["decomposition"]
    ladders = payload["ladders"]
    shared = [r for r in ladders if r["mesh"] == "uniform" and r["steps"] == 16384]
    orders = [r[key] for r in shared for key in ("observed_order_F", "observed_order_Z43")]
    uniform = [r["error_ratio_F_over_Z43"] for r in ladders if r["mesh"] == "uniform"]
    fully_valid = [
        r["error_ratio_F_over_Z43"]
        for r in ladders
        if r["mesh"] == "graded" and r["validity"] == "valid"
    ]
    usable_graded = [
        r["error_ratio_F_over_Z43"]
        for r in ladders
        if r["mesh"] == "graded" and r["validity"] != "indeterminate"
    ]
    methods = [r["method"] for r in rows]

    def two_sig(value):
        result = f"{value:.2g}"
        return result + ".0" if len(result) == 1 and result.isdigit() else result

    def span(values):
        return f"{two_sig(min(values))}–{two_sig(max(values))}"

    table = "\n".join(
        f"| {r['case']} | {r['target']:.0e} | {two_sig(r['method'])} | {two_sig(r['control'])} | "
        f"{two_sig(r['product'])} | {r['target_status']} |"
        for r in rows
    )
    return f"""{HEADING}

On P5 the uniform-baseline factors above mix two effects. With uniform steps both F and Z43 are limited by the noisy initial layer (observed orders {span(orders)} on the finest shared rungs), and their same-step error ratio is only {span(uniform)}, so the uniform "control" factor also carries the grading of that layer. With the γ = 4 graded ladders as the baseline, R(E0b vs F_g) = [F_g/Z43_g] × [Z43_g/E0b]:

| Case | Target | Method F_g/Z43_g | Control Z43_g/E0b | Product | Target status |
| --- | ---: | ---: | ---: | ---: | --- |
{table}

Source: `d3b_graded_decomposition.json`, `/ladders` and `/decomposition`.

On the fully valid graded levels the same-mesh error ratio F_g/Z43_g is {span(fully_valid)}×; including reference-limited levels gives {span(usable_graded)}×. Reading: on P5, adaptive ERK43ZB's advantage over the graded ETDRK4 oracle comes from the method. Once the layer is graded, fixed-step ERK43ZB is about {span(methods)}× cheaper than ETDRK4 at 1e-5, and adaptive ERK43ZB is about as efficient as its own graded oracle. Why ETDRK4 is weaker here is not established. This view changes no D3 verdict.
Source: `d3b_graded_decomposition.json`, `/ladders` and `/decomposition`."""


def render_full():
    base = render_d3b_base(json.loads((RESULTS / "d3b_fixed_pairs.json").read_text()))
    graded = json.loads(OUTPUT.read_text())
    before, marker, after = base.partition(NEXT_HEADING)
    if not marker:
        raise RuntimeError(f"{NEXT_HEADING} missing from {REPORT}")
    before = before.split(HEADING)[0].rstrip()
    return before + "\n\n" + render_section(graded) + "\n\n" + marker + after


def main():
    """Regenerate the committed summaries without running a benchmark."""
    if len(sys.argv) != 1:
        raise SystemExit("usage: python benchmarks/make_d3_summaries.py")
    (RESULTS / "D3_SUMMARY.md").write_text(
        render(json.loads((RESULTS / "d3_work_precision.json").read_text()))
    )
    REPORT.write_text(render_full())


if __name__ == "__main__":
    main()
