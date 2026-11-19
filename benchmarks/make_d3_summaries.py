"""Render D3 and D3b Markdown from committed result files only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT = RESULTS / "d3b_graded_decomposition.json"
REPORT = RESULTS / "D3B_SUMMARY.md"
HEADING = "## Graded two-factor view on P5 (exploratory, post hoc)"
NEXT_HEADING = "## Missing matched-target cells"


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
