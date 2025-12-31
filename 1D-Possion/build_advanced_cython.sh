#!/bin/bash
# Build script for advanced Cython optimizations for LSSVR.

cd /Users/stefanschoder/Downloads/LSSVR/hybrid-FEM-LSSVR/1D-Possion/cython

echo "Building advanced Cython optimizations for LSSVR..."
echo "=================================================="

# Build the advanced Cython module
echo "Building lssvr_optimized.pyx..."
python3 setup_lssvr_optimized.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✓ Advanced Cython optimizations built successfully!"
    echo ""
    echo "Available optimizations:"
    echo "• BLAS-optimized linear algebra operations"
    echo "• Cython iterative refinement"
    echo "• Cython preconditioned conjugate gradients"
    echo "• Batch LSSVR solver framework"
    echo ""
    echo "To use these optimizations, run your LSSVR code normally."
    echo "The system will automatically detect and use the Cython versions."
else
    echo "✗ Failed to build advanced Cython optimizations"
    echo "Make sure you have:"
    echo "• Cython installed: pip install cython"
    echo "• BLAS/LAPACK development libraries"
    echo "• C++ compiler"
    exit 1
fi