#!/usr/bin/env python3
"""
plot_main_figures.py - regenerate every figure in the moljax paper.

    Pavlov, G. "moljax: GPU-accelerated method of lines for stiff
    reaction-diffusion PDEs with FFT preconditioning."
    Computer Physics Communications 326 (2026) 110205.
    doi:10.1016/j.cpc.2026.110205

Usage
-----
    python3 benchmarks/plot_main_figures.py            # all figures
    python3 benchmarks/plot_main_figures.py --list     # show what would run
    python3 benchmarks/plot_main_figures.py --only pattern_gallery attractor
    python3 benchmarks/plot_main_figures.py --skip-slow

Note on cost
------------
This is a figure *driver*, not a pure plotter. Several of the paper's
figure generators recompute the trajectories they plot rather than
reading cached JSON from ``results/`` (the work-precision sweeps and the
pattern/attractor galleries in particular integrate their own reference
solutions). Running this script therefore repeats that computation; it is
not a cheap post-processing pass over ``run_all.sh`` output. Use
``--skip-slow`` to restrict the run to the generators that finish in
seconds to a couple of minutes.

Figures are written next to each generator's configured output path,
normally ``figures/`` at the repository root, in both PDF and PNG.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# (key, description, script, slow, figures produced)
FIGURES: list[tuple[str, str, str, bool, tuple[str, ...]]] = [
    (
        "convergence",
        "Convergence verification (spatial + temporal)",
        "generate_convergence_figure.py",
        False,
        ("convergence.pdf",),
    ),
    (
        "reactor_steep_gradients",
        "Reactor concentration profiles across Pe regimes",
        "generate_reactor_steep_gradients.py",
        False,
        ("fig_reactor_steep_gradients.pdf", "fig_reactor_pe_progression.pdf"),
    ),
    (
        "reactor_method_comparison",
        "Reactor method comparison (RK4 / CN / IMEX)",
        "benchmark_method_comparison.py",
        False,
        ("fig_reactor_method_comparison.pdf", "fig_diffusion_work_precision.pdf"),
    ),
    (
        "gmres_sweep",
        "GMRES iterations vs stiffness ratio",
        "benchmark_gmres_sweep.py",
        False,
        ("gmres_sweep.pdf",),
    ),
    (
        "scaling",
        "Scaling with problem size (CPU vs GPU)",
        "benchmark_scaling.py",
        False,
        ("scaling.pdf",),
    ),
    (
        "diffrax_work_precision",
        "Work-precision: FFT-CN vs Diffrax Tsit5",
        "benchmark_diffrax_work_precision.py",
        False,
        ("fig_diffrax_work_precision.pdf",),
    ),
    (
        "work_precision_reactor",
        "Work-precision: reactor (NFE)",
        "work_precision_reactor_nfe.py",
        False,
        ("fig_work_precision_reactor_nfe.pdf",),
    ),
    (
        "pattern_gallery",
        "Turing pattern gallery (Gray-Scott / Schnakenberg / Brusselator)",
        "generate_pattern_figures.py",
        True,
        ("fig_pattern_gallery.pdf",),
    ),
    (
        "attractor",
        "Attractor divergence (3 systems x 3 integrators)",
        "generate_attractor_figure.py",
        True,
        ("fig_attractor_divergence.pdf",),
    ),
    (
        "work_precision_gray_scott",
        "Work-precision: Gray-Scott",
        "work_precision_gray_scott.py",
        True,
        ("fig_work_precision_gray_scott.pdf",),
    ),
    (
        "work_precision_brusselator",
        "Work-precision: Brusselator",
        "work_precision_brusselator.py",
        True,
        ("fig_work_precision_brusselator.pdf",),
    ),
    (
        "work_precision_schnakenberg",
        "Work-precision: Schnakenberg (and diffusion WP)",
        "generate_all_wp_nfe.py",
        True,
        ("fig_work_precision_schnakenberg.pdf",),
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate the moljax paper figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="list figures and exit")
    p.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="run only these figure keys",
    )
    p.add_argument(
        "--skip-slow",
        action="store_true",
        help="skip generators that recompute long trajectories",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter used for the generator scripts",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    selected = FIGURES
    if args.only:
        keys = set(args.only)
        unknown = keys - {f[0] for f in FIGURES}
        if unknown:
            print(f"Unknown figure keys: {sorted(unknown)}", file=sys.stderr)
            print(f"Available: {[f[0] for f in FIGURES]}", file=sys.stderr)
            return 2
        selected = [f for f in FIGURES if f[0] in keys]
    if args.skip_slow:
        selected = [f for f in selected if not f[3]]

    if args.list:
        width = max(len(f[0]) for f in FIGURES)
        for key, desc, script, slow, figs in FIGURES:
            tag = "slow" if slow else "    "
            print(f"  [{tag}] {key:<{width}}  {desc}")
            print(f"         {'':<{width}}  {script} -> {', '.join(figs)}")
        return 0

    missing = [f for f in selected if not (BENCH_DIR / f[2]).exists()]
    if missing:
        print("Missing generator scripts:", file=sys.stderr)
        for f in missing:
            print(f"  {f[2]} (for {f[0]})", file=sys.stderr)
        return 2

    passed: list[str] = []
    failed: list[str] = []

    for key, desc, script, slow, figs in selected:
        print()
        print("=" * 70)
        print(f">>> {desc}")
        print(f"    {script}" + ("  [slow]" if slow else ""))
        print("=" * 70)
        start = time.time()
        proc = subprocess.run([args.python, script], cwd=BENCH_DIR)
        elapsed = time.time() - start
        if proc.returncode == 0:
            print(f"--- OK ({elapsed:.1f}s): {', '.join(figs)}")
            passed.append(key)
        else:
            print(
                f"--- FAILED ({elapsed:.1f}s, exit {proc.returncode}): {key}",
                file=sys.stderr,
            )
            failed.append(key)

    print()
    print("=" * 70)
    print(f"Figures complete. Passed: {len(passed)}  Failed: {len(failed)}")
    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
