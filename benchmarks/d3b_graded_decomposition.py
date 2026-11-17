"""Exploratory, post-hoc graded P5 decomposition from committed D3 JSONs.

Run ``JAX_PLATFORMS=cpu PYTHONPATH="$PWD" python benchmarks/d3b_graded_decomposition.py``.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import jax

from benchmarks.d3_work_precision_analysis import TARGETS, groups, interpolate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
BASE_COMMIT = "009fe911824798a679c30e3120e3e9e999c69572"
INPUTS = {
    "d3_work_precision.json": "cd6a166980c436df5d6abdf1cc4fd10a7e85914b04ba49b97a65d77a349e2910",
    "d3b_fixed_pairs.json": "8e24c53106eb47dce9d6c08eb5c3b4df73c06df557d4d05f42c061701b0f6b42",
}
OUTPUT = RESULTS / "d3b_graded_decomposition.json"
REPORT = RESULTS / "D3B_SUMMARY.md"
HEADING = "## Graded two-factor view on P5 (exploratory, post hoc)"
NEXT_HEADING = "## Missing matched-target cells"


def load_inputs():
    payloads = {}
    for name, expected in INPUTS.items():
        content = (RESULTS / name).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name}: SHA256 {actual} differs from {expected}")
        payloads[name] = json.loads(content)
    return payloads


def combined_validity(a, b):
    if "indeterminate" in (a, b):
        return "indeterminate"
    if "valid, reference-limited" in (a, b):
        return "valid, reference-limited"
    return "valid"


def render():
    inputs = load_inputs()
    d3 = inputs["d3_work_precision.json"]
    d3b = inputs["d3b_fixed_pairs.json"]
    grouped = groups(d3["records"] + d3b["ladders"])
    epsilon = {(r["problem"], r["N"]): r["epsilon_conf"] for r in d3["reference_confirmation"]}
    decomposition, ladders = [], []

    for n in (64, 128):
        for target in TARGETS:
            costs = {
                config: interpolate(grouped["P5", n, config], target)
                for config in ("F_g", "Z43_g", "E0b")
            }
            if any(value is None for value in costs.values()):
                continue
            method = costs["F_g"] / costs["Z43_g"]
            control = costs["Z43_g"] / costs["E0b"]
            product = method * control
            direct = costs["F_g"] / costs["E0b"]
            if abs(product / direct - 1) > 1e-12:
                raise RuntimeError(f"P5/{n}, {target}: decomposition identity fails")
            decomposition.append(
                {
                    "case": f"P5/{n}",
                    "target": target,
                    "costs_NFE": costs,
                    "method": method,
                    "control": control,
                    "product": product,
                    "R_direct": direct,
                    "target_status": "confirmed"
                    if target >= 10 * epsilon["P5", n]
                    else "self-convergence only",
                }
            )

        for level in range(5):
            f = next(r for r in grouped["P5", n, "F_g"] if r["level"] == level)
            z = next(r for r in grouped["P5", n, "Z43_g"] if r["level"] == level)
            ladders.append(
                {
                    "case": f"P5/{n}",
                    "mesh": "graded",
                    "level": level,
                    "steps": f["steps"],
                    "error_ratio_F_over_Z43": f["error_inf_relative"] / z["error_inf_relative"],
                    "validity": combined_validity(f["validity"], z["validity"]),
                    "F_validity": f["validity"],
                    "Z43_validity": z["validity"],
                }
            )

        fixed = {
            config: {r["steps"]: r for r in grouped["P5", n, config] if r.get("success")}
            for config in ("F", "Z43")
        }
        for steps in sorted(fixed["F"].keys() & fixed["Z43"].keys()):
            f, z = fixed["F"][steps], fixed["Z43"][steps]
            ladders.append(
                {
                    "case": f"P5/{n}",
                    "mesh": "uniform",
                    "steps": steps,
                    "error_ratio_F_over_Z43": f["error_inf_relative"] / z["error_inf_relative"],
                    "observed_order_F": f.get("observed_order"),
                    "observed_order_Z43": z.get("observed_order"),
                    "validity": combined_validity(f["validity"], z["validity"]),
                    "F_validity": f["validity"],
                    "Z43_validity": z["validity"],
                }
            )
        for config in ("F", "Z43"):
            for row in sorted(fixed[config].values(), key=lambda item: item["steps"]):
                if row["validity"] != "indeterminate":
                    ladders.append(
                        {
                            "case": f"P5/{n}",
                            "mesh": "uniform_order",
                            "config": config,
                            "steps": row["steps"],
                            "observed_order": row.get("observed_order"),
                            "validity": row["validity"],
                        }
                    )

    expected = {
        "P5/64": (3.48, 0.969, 3.37, "self-convergence only"),
        "P5/128": (2.96, 0.957, 2.83, "confirmed"),
    }
    for case, (method, control, product, status) in expected.items():
        row = next(r for r in decomposition if (r["case"], r["target"]) == (case, 1e-5))
        if (
            abs(row["method"] - method) > 0.02
            or abs(row["control"] - control) > 0.005
            or abs(row["product"] - product) > 0.02
            or row["target_status"] != status
        ):
            raise RuntimeError(f"{case}: expected-value gate fails: {row}")

    return {
        "provenance": {
            "label": "exploratory, post hoc",
            "input_sha256": INPUTS,
            "base_commit": BASE_COMMIT,
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "analysis_helper": "benchmarks.d3_work_precision_analysis.interpolate",
        },
        "decomposition": decomposition,
        "ladders": ladders,
    }


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
Source: `d3b_graded_decomposition.json`, `/ladders`.

| Case | Target | Method F_g/Z43_g | Control Z43_g/E0b | Product | Target status |
| --- | ---: | ---: | ---: | ---: | --- |
{table}
Source: `d3b_graded_decomposition.json`, `/decomposition`.

On the fully valid graded levels the same-mesh error ratio F_g/Z43_g is {span(fully_valid)}×; including reference-limited levels gives {span(usable_graded)}×. Reading: on P5, adaptive ERK43ZB's advantage over the graded ETDRK4 oracle comes from the method. Once the layer is graded, fixed-step ERK43ZB is about {span(methods)}× cheaper than ETDRK4 at 1e-5, and adaptive ERK43ZB is about as efficient as its own graded oracle. Why ETDRK4 is weaker here is not established. This view changes no D3 verdict.
Source: `d3b_graded_decomposition.json`, `/ladders` and `/decomposition`."""


def write_report(payload):
    old = REPORT.read_text()
    before, marker, after = old.partition(NEXT_HEADING)
    if not marker:
        raise RuntimeError(f"{NEXT_HEADING} missing from {REPORT}")
    before = before.split(HEADING)[0].rstrip()
    REPORT.write_text(before + "\n\n" + render_section(payload) + "\n\n" + marker + after)


if __name__ == "__main__":
    payload = render()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload)
    print(OUTPUT)
