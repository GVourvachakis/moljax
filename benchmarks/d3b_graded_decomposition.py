"""Exploratory, post-hoc graded P5 decomposition from committed D3 JSONs.

Run ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=. .venv/bin/python benchmarks/d3b_graded_decomposition.py``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import jax

from benchmarks.d3_work_precision_analysis import TARGETS, groups, interpolate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
INPUTS = ("d3_work_precision.json", "d3b_fixed_pairs.json")
OUTPUT = RESULTS / "d3b_graded_decomposition.json"
REPORT = RESULTS / "D3B_SUMMARY.md"


def load_inputs():
    payloads = {}
    for name in INPUTS:
        content = (RESULTS / name).read_bytes()
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
    epsilon = {
        (case.split("_")[0], int(case.split("_")[1])): detail["epsilon_conf"]
        for case, detail in d3["reference_confirmation"].items()
    }
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
            "input_sha256": {
                name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in INPUTS
            },
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "analysis_helper": "benchmarks.d3_work_precision_analysis.interpolate",
        },
        "decomposition": decomposition,
        "ladders": ladders,
    }




def main():
    payload = render()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    from benchmarks.make_d3_summaries import render_full

    REPORT.write_text(render_full())
    print(OUTPUT)


if __name__ == "__main__":
    main()
