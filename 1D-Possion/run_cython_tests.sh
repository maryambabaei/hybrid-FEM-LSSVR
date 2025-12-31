#!/bin/bash
# Convenience script to run Cython tests and benchmarks

set -e  # Exit on any error

echo "Advanced Cython Optimizations - Test & Benchmark Suite"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "cython/test_cython.py" ]; then
    echo "Error: Please run this script from the 1D-Possion directory"
    exit 1
fi

# Check if Cython extensions are built
if [ ! -f "cython/lssvr_optimized.cpython-*.so" ] && [ ! -f "cython/lssvr_optimized.so" ]; then
    echo "Warning: Cython extensions not found. Building..."
    ./build_advanced_cython.sh
fi

echo ""
echo "Running Cython functionality tests..."
python3 cython/test_cython.py

echo ""
echo "Running comprehensive benchmarks..."
python3 cython/benchmark_cython.py

echo ""
echo "All tests and benchmarks completed successfully!"
echo ""
echo "Next steps:"
echo "1. Check the benchmark results above for performance improvements"
echo "2. Use the optimizations in your LSSVR code (they load automatically)"
echo "3. For production use, consider the batch processing framework"