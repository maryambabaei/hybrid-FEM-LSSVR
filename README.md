## Higher-resolution-LSSVR

This repository presents a LSSVR approach for super-resolution of elliptic PDE solutions. A coarse solution is first computed using the Finite Element Method, then Least Squares Support Vector Regression is used to reconstruct a higher-resolution solution from the low-resolution data. The approach is demonstrated on one-dimensional elliptic problems, including the Poisson equation.

### Problem Description

This code solves the 1D Poisson equation:

-u''(x) = π² sin(πx),  x ∈ [-1, 1]

with homogeneous Dirichlet boundary conditions:

u(-1) = 0
u(1) = 0

The exact solution is: u(x) = sin(πx)

### Method Overview

The hybrid method works in three steps:

FEM Solution: Solve the problem using standard FEM to get nodal values
LSSVR Enhancement: Improve the solution inside each element using LSSVR with Legendre polynomial kernels while keeping FEM nodal values fixed
Global Solution: Combine all element solutions to get the final enhanced solution

### Requirements

Install the required packages:
```
pip install scikit-fem[all]
```
Package Versions Used

numpy - for numerical computations
scipy - for optimization routines
matplotlib - for visualization
scikit-fem (version 11.0.0) - for FEM implementation and evaluation
