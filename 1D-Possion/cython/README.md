# Cython Build Files

This directory contains the Cython implementation for optimized Legendre matrix computations.

## Files

- `legendre_matrices.pyx`: Cython source code for Legendre polynomial computations
- `legendre_matrices.c`: Generated C code from Cython compilation
- `setup_cython.py`: Setup script for building the Cython extension
- `build/`: Build artifacts directory

## Building

To rebuild the Cython extension, run the build script from the parent directory:

```bash
./build_cython.sh
```

This will:
1. Change to the cython directory
2. Compile the Cython code using the virtual environment
3. Move the compiled `.so` file back to the main directory for importing

## Requirements

- Cython (already installed in the virtual environment)
- NumPy development headers
- C compiler (clang on macOS)

## Notes

The compiled extension (`legendre_matrices_cython.cpython-*-darwin.so`) is kept in the main directory for easy importing by the Python code.