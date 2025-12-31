#!/bin/bash
# Build script for Cython extension
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/cython"
# Use virtual environment Python
PYTHON="$SCRIPT_DIR/../.venv/bin/python"
$PYTHON setup_cython.py build_ext --inplace
# Move the compiled .so file back to the main directory
mv legendre_matrices_cython*.so ../
cd ..
echo "Cython extension built successfully"