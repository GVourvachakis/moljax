"""Checks for the committed D3 and D3b Markdown renderers."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
DOCUMENTS = ("D3_SUMMARY.md", "D3B_SUMMARY.md")
INPUTS = (
    "d3_work_precision.json",
    "d3b_fixed_pairs.json",
    "d3b_graded_decomposition.json",
    "d3_references.json",
    "d3_timing_study_session.json",
    "d3_reproduction.json",
)


def _matches(data, pointer):
    nodes = [data]
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        next_nodes = []
        for node in nodes:
            if key == "*":
                next_nodes.extend(node.values() if isinstance(node, dict) else node)
            elif isinstance(node, dict) and key in node:
                next_nodes.append(node[key])
            elif isinstance(node, list) and key.isdecimal() and int(key) < len(node):
                next_nodes.append(node[int(key)])
        nodes = next_nodes
        if not nodes:
            return False
    return bool(nodes)


def test_rendered_summaries_are_committed_and_idempotent():
    code = (
        "import json; from pathlib import Path; "
        "from benchmarks.make_d3_summaries import render, render_full, RESULTS; "
        "p=json.loads((RESULTS/'d3_work_precision.json').read_text()); "
        "a=[render(p),render_full()]; b=[render(p),render_full()]; "
        "assert a==b; print(json.dumps(a))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        check=True,
    )
    rendered = json.loads(proc.stdout)
    for name, value in zip(DOCUMENTS, rendered, strict=True):
        assert value == (RESULTS / name).read_text()


def test_every_pointer_resolves_and_table_sources_are_separate():
    inputs = [json.loads((RESULTS / name).read_text()) for name in INPUTS]
    for name in DOCUMENTS:
        lines = (RESULTS / name).read_text().splitlines()
        for line, following in zip(lines, lines[1:], strict=False):
            if line.startswith("|") and line.endswith("|"):
                assert not following or following.startswith("|"), (name, line, following)
        pointers = re.findall(r"`(/[^`]+)`", "\n".join(lines))
        assert pointers
        for pointer in pointers:
            assert any(_matches(data, pointer) for data in inputs), (name, pointer)
