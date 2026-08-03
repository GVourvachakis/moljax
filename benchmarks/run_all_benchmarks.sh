#!/bin/bash
#
# Deprecated: kept so existing invocations keep working.
#
# The canonical entry point is benchmarks/run_all.sh, which is the name
# used in the paper's reproduction quickstart and which covers the full
# benchmark suite. This older script omitted the Schnakenberg and
# Brusselator systems, the work-precision sweeps, and the OFAT, ablation,
# CuPy-FFT and JIT-factorial studies.
#
echo "note: run_all_benchmarks.sh is deprecated; running run_all.sh instead." >&2
exec bash "$(dirname "$0")/run_all.sh" "$@"
