"""Compare committed D3/D3b rows with the original study, without solving.

Get inputs with ``git show f3a05c9:benchmarks/results/d3_work_precision.json > study.json``
and ``git show f3a05c9:benchmarks/results/d3b_fixed_pairs.json > study_d3b.json``.
Run ``python benchmarks/d3_reproduction.py study.json study_d3b.json``.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
STUDY_COMMIT = "f3a05c93a07a2cf3684660f94f410b989bd119e6"
D3_FIELDS = (
    "accepted",
    "rejected",
    "attempts",
    "NFE",
    "transforms",
    "status",
    "endpoint_sha256",
    "error_inf_relative",
    "error_inf_absolute",
    "error_inf_relative_per_field",
    "validity",
)
D3B_FIELDS = (
    "steps",
    "NFE",
    "endpoint_sha256",
    "error_inf_absolute",
    "error_inf_relative",
    "error_inf_per_field",
    "error_inf_relative_per_field",
    "status",
    "validity",
)


def key(row):
    config = row["config"]
    if config == "Z43" and row.get("level") is not None:
        config = "Z43_g"
    value = (
        row["level"]
        if row.get("level") is not None
        else row["steps"]
        if row.get("steps") is not None
        else row.get("rtol")
    )
    return row["problem"], row["N"], config, value


def compare(source, current, fields):
    baseline = {key(row): row for row in source}
    if len(baseline) != len(source):
        raise RuntimeError("study contains duplicate row keys")
    identical = 0
    for row in current:
        old = baseline[key(row)]
        if all(row.get(field) == old.get(field) for field in fields):
            identical += 1
    return identical


def main(study_path, study_d3b_path):
    study = json.loads(study_path.read_text())
    study_d3b = json.loads(study_d3b_path.read_text())
    here = json.loads((RESULTS / "d3_work_precision.json").read_text())
    here_d3b = json.loads((RESULTS / "d3b_fixed_pairs.json").read_text())
    d3_count = compare(study["records"], here["records"], D3_FIELDS)
    d3b_count = compare(study_d3b["ladders"], here_d3b["ladders"], D3B_FIELDS)
    measured = [row for row in here_d3b["ladders"] if row["config"] not in ("F", "F_g")]
    measured_identical = compare(study_d3b["ladders"], measured, D3B_FIELDS)
    output = {
        "study_commit": STUDY_COMMIT,
        "study_json_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
        "study_d3b_json_sha256": hashlib.sha256(study_d3b_path.read_bytes()).hexdigest(),
        "d3": {
            "fields_compared": D3_FIELDS,
            "records_compared": len(here["records"]),
            "records_identical": d3_count,
        },
        "d3b": {
            "fields_compared": D3B_FIELDS,
            "measured_rows_compared": len(measured),
            "measured_rows_identical": measured_identical,
            "all_rows_compared": len(here_d3b["ladders"]),
            "all_rows_identical": d3b_count,
        },
    }
    if (d3_count, measured_identical, d3b_count) != (
        len(here["records"]),
        len(measured),
        len(here_d3b["ladders"]),
    ):
        raise RuntimeError("study and branch records differ")
    (RESULTS / "d3_reproduction.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python benchmarks/d3_reproduction.py study.json study_d3b.json")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
