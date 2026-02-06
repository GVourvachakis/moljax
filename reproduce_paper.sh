#!/bin/bash
#
# reproduce_paper.sh — Single script to reproduce ALL results from:
#
#   "Accelerating Stiff Reaction-Diffusion Simulation:
#    GPU Method of Lines with FFT Preconditioning in JAX"
#
#   Gorgi Pavlov, 2025. Submitted to Journal of Computational Physics.
#
# USAGE:
#   ./reproduce_paper.sh          # Full run (~60-90 min on RTX 5060)
#   ./reproduce_paper.sh --quick  # Quick validation (~5 min, fewer reps)
#
# PREREQUISITES:
#   1. NVIDIA GPU with CUDA support (8+ GB VRAM)
#   2. Python 3.10+ with JAX[cuda12], numpy, scipy, matplotlib, diffrax
#   3. moljax installed:  pip install -e .
#
# See REPRODUCE.md for detailed setup instructions.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/benchmarks"
RESULTS_DIR="$BENCH_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"

# Parse arguments
QUICK=false
if [[ "$1" == "--quick" ]]; then
    QUICK=true
    REPS_FLAG="--n-reps 3"
    echo "=== QUICK MODE: 3 reps per benchmark (validation only) ==="
else
    REPS_FLAG=""
fi

echo "============================================================"
echo "  moljax Paper Reproduction Script"
echo "============================================================"
echo "Started: $(date)"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ---- Step 0: Environment check ----
echo "--- Environment Check ---"
python3 -c "
import jax
jax.config.update('jax_enable_x64', True)
print(f'JAX version:  {jax.__version__}')
print(f'Devices:      {jax.devices()}')
print(f'Backend:      {jax.default_backend()}')
print(f'Float64:      {jax.config.jax_enable_x64}')
import numpy, scipy, matplotlib
print(f'NumPy:        {numpy.__version__}')
print(f'SciPy:        {scipy.__version__}')
print(f'Matplotlib:   {matplotlib.__version__}')
try:
    import diffrax; print(f'Diffrax:      {diffrax.__version__}')
except ImportError:
    print('Diffrax:      NOT INSTALLED (Diffrax benchmarks will be skipped)')
"
echo ""

# ---- Step 1: Create output directories ----
mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

cd "$BENCH_DIR"

# ---- Step 2: Core benchmarks (Tables 5-8, Figures 6-14) ----

echo "============================================================"
echo "  1/10  SciPy Comparison (Table 5, Figure 6)"
echo "============================================================"
python3 benchmark_vs_scipy.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  2/10  CPU/GPU Scaling (Figure 7)"
echo "============================================================"
python3 benchmark_scaling.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  3/10  Gray-Scott Pattern Formation (Table 6)"
echo "============================================================"
python3 benchmark_gray_scott.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  4/10  GMRES Iteration Sweep (Table 7, Figure 8)"
echo "============================================================"
python3 benchmark_gmres_sweep.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  5/10  JIT Compilation Speedup (Figure 9)"
echo "============================================================"
python3 benchmark_jit_speedup.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  6/10  Stiff ADR Benchmark / Tubular Reactor (Table 8)"
echo "============================================================"
python3 benchmark_tubular_reactor.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  7/10  Method Comparison (Figure 10)"
echo "============================================================"
python3 benchmark_method_comparison.py $REPS_FLAG

echo ""
echo "============================================================"
echo "  8/10  Solver Comparison (Figure 11)"
echo "============================================================"
python3 benchmark_solver_comparison.py $REPS_FLAG

# ---- Step 3: Broadening / stress test experiments ----

echo ""
echo "============================================================"
echo "  9/10  Variable-Coefficient Stress Test (Table 9, Figure 12)"
echo "============================================================"
python3 benchmark_variable_coeff.py $REPS_FLAG

echo ""
echo "============================================================"
echo " 10/10  Mixed Boundary Conditions (Table 10, Figure 13)"
echo "============================================================"
python3 benchmark_mixed_bc.py $REPS_FLAG

# ---- Step 4: Generate all paper figures ----

echo ""
echo "============================================================"
echo "  Generating paper figures"
echo "============================================================"
python3 generate_paper_figures.py

# ---- Step 5: Verify outputs ----

echo ""
echo "============================================================"
echo "  Verification"
echo "============================================================"

echo "Results JSON files:"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

echo ""
echo "Figures:"
ls -lh "$FIGURES_DIR"/fig_*.{pdf,png} 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}' || true

echo ""
echo "--- Quick sanity check ---"
python3 -c "
import json
from pathlib import Path

rd = Path('$RESULTS_DIR')

# Check GMRES sweep
with open(rd / 'gmres_sweep.json') as f:
    g = json.load(f)
best = max(r['reduction'] for r in g['results'])
print(f'Max GMRES reduction (FFT precond): {best:.0f}x')

# Check variable-coeff
with open(rd / 'variable_coeff.json') as f:
    v = json.load(f)
step_1000 = [r for r in v['results'] if r['profile']=='step' and r['contrast']==1000.0]
if step_1000:
    print(f'Variable-coeff step contrast=1000: {step_1000[0][\"reduction\"]:.1f}x reduction')

# Check mixed BC
with open(rd / 'mixed_bc.json') as f:
    m = json.load(f)
n256 = [r for r in m['results'] if r['N']==256]
if n256:
    sp = n256[0]['speedup_fft_over_fd']
    print(f'Mixed BC FFT-DST speedup at 256x256: {sp:.1f}x')
"

echo ""
echo "============================================================"
echo "  DONE"
echo "============================================================"
echo "Finished: $(date)"
echo ""
echo "All results in:  $RESULTS_DIR/"
echo "All figures in:   $FIGURES_DIR/"
echo ""
echo "Compare results against paper claims in REPRODUCE.md"
