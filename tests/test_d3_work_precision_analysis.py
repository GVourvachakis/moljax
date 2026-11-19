"""Pure work-precision checks for the D3 benchmark analysis."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_reference_archive_matches_manifest():
    manifest = json.loads((ROOT / "benchmarks/results/d3_references.json").read_text())
    with np.load(ROOT / "benchmarks/results/d3_references.npz") as archive:
        assert len(manifest["cases"]) == 6
        for case, detail in manifest["cases"].items():
            for field, expected in detail["arrays"].items():
                assert hashlib.sha256(archive[f"{case}_{field}"].tobytes()).hexdigest() == expected


def test_pareto_interpolation_excludes_invalid_and_never_extrapolates():
    # The benchmark enables JAX x64 on import; keep that process-wide change isolated.
    source = """
from benchmarks.d3_work_precision_analysis import interpolate, pareto

def row(error, work, validity="valid"):
    return {"success": True, "validity": validity, "error_inf_relative": error, "NFE": work}

rows = [
    row(1e-5, 100), row(1e-4, 50), row(1e-3, 25),
    row(5e-5, 150), row(1e-6, 200, "indeterminate"), row(2e-2, 2),
]
assert [r["NFE"] for r in pareto(rows)] == [100, 50, 25]
assert interpolate(rows, 1e-4) == 50
assert interpolate(rows, 1e-2) is None
assert interpolate(rows, 1e-6) is None
assert abs(interpolate(rows, 1e-4 * 10**0.5) - 50 / 2**0.5) < 1e-12
"""
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
