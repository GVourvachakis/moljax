#!/bin/bash
#
# Run ALL moljax benchmarks and generate ALL paper figures
#
# Expected runtime: 60-90 minutes on RTX 4090
#                   2-3 hours on RTX 3060
#                   8-12 hours on CPU only
#

set -e  # Exit on error

echo "=========================================="
echo "moljax Benchmark Suite"
echo "=========================================="
echo "Started: $(date)"
echo ""

# Check GPU availability
python -c "import jax; print(f'Devices: {jax.devices()}'); print(f'Default backend: {jax.default_backend()}')"
echo ""

# Core benchmarks
echo "=========================================="
echo "1/10: SciPy Comparison"
echo "=========================================="
python benchmark_vs_scipy.py

echo ""
echo "=========================================="
echo "2/10: CPU/GPU Scaling"
echo "=========================================="
python benchmark_scaling.py

echo ""
echo "=========================================="
echo "3/10: Gray-Scott Pattern Formation"
echo "=========================================="
python benchmark_gray_scott.py

echo ""
echo "=========================================="
echo "4/10: GMRES Iteration Sweep"
echo "=========================================="
python benchmark_gmres_sweep.py

echo ""
echo "=========================================="
echo "5/10: JIT Speedup"
echo "=========================================="
python benchmark_jit_speedup.py

echo ""
echo "=========================================="
echo "6/10: Tubular Reactor Case Study"
echo "=========================================="
python benchmark_tubular_reactor.py

echo ""
echo "=========================================="
echo "7/10: Method Comparison"
echo "=========================================="
python benchmark_method_comparison.py

echo ""
echo "=========================================="
echo "8/10: Solver Comparison"
echo "=========================================="
python benchmark_solver_comparison.py

# Broadening experiments
echo ""
echo "=========================================="
echo "9/10: Variable-Coefficient Stress Test"
echo "=========================================="
python benchmark_variable_coeff.py

echo ""
echo "=========================================="
echo "10/10: Mixed Boundary Conditions"
echo "=========================================="
python benchmark_mixed_bc.py

# Generate all paper figures
echo ""
echo "=========================================="
echo "Generating Paper Figures"
echo "=========================================="
python generate_paper_figures.py

echo ""
echo "=========================================="
echo "Benchmark Suite Complete"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to: benchmarks/results/"
echo "Figures saved to: figures/"
echo ""
echo "Next steps:"
echo "  1. Check results: ls results/*.json"
echo "  2. Check figures: ls ../figures/*.pdf"
echo "  3. Verify key metrics against REPRODUCE.md"
