"""Compare committed D3/D3b rows with the original study, without solving.

From an upstream clone, first run
``git fetch https://github.com/GVourvachakis/moljax experiment/etd-embedded-pairs feature/etd-embedded-pairs``.
Get inputs with ``git show f3a05c9:benchmarks/results/d3_work_precision.json > study.json``
and ``git show f3a05c9:benchmarks/results/d3b_fixed_pairs.json > study_d3b.json``.
For the runner-refactor check, also get
``git show deb96e7:benchmarks/d3_work_precision.py > runner_measured.py``.
Run ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=. python benchmarks/d3_reproduction.py study.json study_d3b.json runner_measured.py``.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
STUDY_COMMIT = "f3a05c93a07a2cf3684660f94f410b989bd119e6"
RUNNER_BLOB_COMMIT = "deb96e7de9fc54d0e67549babfdb6392061dbeda"
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


def runner_refactor(measured_path, measurement):
    """Compare all setup inputs from the recorded runner with the current one."""
    import jax
    import numpy as np

    from benchmarks import d3_work_precision as current

    current.check_environment()
    measured_hash = hashlib.sha256(measured_path.read_bytes()).hexdigest()
    if measured_hash != measurement["script_sha256"]:
        raise RuntimeError("measurement-time runner checksum differs from the D3 record")
    spec = importlib.util.spec_from_file_location("d3_runner_measured", measured_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the measurement-time runner")
    measured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measured)
    cases = []
    for name, n in current.CASES:
        y_old, ops_old, rhs_old, _, final_old = measured.setup(name, n)
        y_new, ops_new, rhs_new, _, final_new = current.setup(name, n)
        if np.asarray(final_old).tobytes() != np.asarray(final_new).tobytes():
            raise RuntimeError(f"runner final time differs for {name}/{n}")
        fingerprints = {"y0_sha256": current.endpoint_sha(y_new)}
        if current.endpoint_sha(y_old) != fingerprints["y0_sha256"]:
            raise RuntimeError(f"runner initial state differs for {name}/{n}")
        fingerprints["rhs_sha256"] = {}
        for label, t in (("t0", 0.0), ("t037", 0.37), ("final", final_new)):
            digest = current.endpoint_sha(rhs_new(y_new, t))
            if current.endpoint_sha(rhs_old(y_old, t)) != digest:
                raise RuntimeError(f"runner RHS differs for {name}/{n} at {label}")
            fingerprints["rhs_sha256"][label] = digest
        fingerprints["operator_sha256"] = {}
        if ops_old.keys() != ops_new.keys():
            raise RuntimeError(f"runner operator fields differ for {name}/{n}")
        for field in ops_new:
            digest = current.endpoint_sha({field: ops_new[field].matvec(y_new[field])})
            if current.endpoint_sha({field: ops_old[field].matvec(y_old[field])}) != digest:
                raise RuntimeError(f"runner operator action differs for {name}/{n}/{field}")
            fingerprints["operator_sha256"][field] = digest
        cases.append({"case": f"{name}/{n}", "final_time": final_new, **fingerprints})
    return {
        "measurement_head_commit": measurement["git_commit"],
        "runner_blob_commit": RUNNER_BLOB_COMMIT,
        "measured_runner_sha256": measured_hash,
        "current_runner_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "all_identical": True,
        "jax": jax.__version__,
        "numpy": np.__version__,
    }


def main(study_path, study_d3b_path, measured_path=None):
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
    if measured_path is not None:
        output["runner_refactor"] = runner_refactor(
            measured_path, here["provenance"]["measurement"]
        )
    (RESULTS / "d3_reproduction.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "usage: python benchmarks/d3_reproduction.py study.json study_d3b.json [runner_measured.py]"
        )
    main(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]) if len(sys.argv) == 4 else None,
    )
