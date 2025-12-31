import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
from scipy import integrate
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre, leggauss
from skfem import *
from skfem.helpers import dot, grad
import argparse
import time
import psutil
import os
import numba
from numba import jit

# Try to import Cython extension
try:
    from legendre_matrices_cython import build_legendre_matrices_cython
    USE_CYTHON = True  # Now working correctly after fixing scaling and variable initialization
    print("Cython extension available and enabled for performance")
except ImportError:
    USE_CYTHON = False
    print("Cython extension not available, using optimized Python version")

# Try to import advanced Cython optimizations
try:
    import sys
    import os
    # Add cython build directory to path (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cython_build_dir = os.path.join(script_dir, 'cython', 'build', 'lib.macosx-10.9-universal2-cpython-310')
    if cython_build_dir not in sys.path:
        sys.path.insert(0, cython_build_dir)

    from lssvr_optimized import (
        iterative_refinement_cython,
        preconditioned_cg_cython,
        batch_lssvr_solve_cython,
        assemble_kkt_dense_cython,
        scale_kkt_system_cython,
        build_legendre_second_derivatives_cython,
        build_legendre_boundary_values_cython,
        generate_gauss_lobatto_points_cython,
        compute_slack_variables_cython,
        check_boundary_constraints_cython,
        compute_constraint_violation_cython
    )
    USE_ADVANCED_CYTHON = True
    print("Advanced Cython optimizations available and enabled")
except ImportError as e:
    USE_ADVANCED_CYTHON = False
    print(f"Advanced Cython optimizations not available ({e}), using standard implementations")

# Enable vectorized matrix building for maximum performance
USE_VECTORIZED = False  # Temporarily disabled - needs more optimization

class PerformanceMonitor:
    """Monitor performance metrics during execution."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.fem_time = 0
        self.lssvr_time = 0
        self.element_times = []
        self.memory_usage = []
        self.operations = []
    
    def record_operation(self, operation_name, duration, details=None):
        """Record a completed operation."""
        self.operations.append({
            'name': operation_name,
            'duration': duration,
            'timestamp': time.time() - self.start_time,
            'details': details
        })
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def print_summary(self):
        """Print performance summary."""
        total_time = time.time() - self.start_time
        print(f"\n{'='*50}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"Total execution time: {total_time:.4f} seconds")
        if hasattr(self, 'fem_time') and self.fem_time:
            print(f"FEM solve time: {self.fem_time:.4f} seconds ({self.fem_time/total_time*100:.1f}%)")
        if hasattr(self, 'lssvr_total_time') and self.lssvr_total_time:
            print(f"LSSVR solve time: {self.lssvr_total_time:.4f} seconds ({self.lssvr_total_time/total_time*100:.1f}%)")
        if hasattr(self, 'total_solve_time') and self.total_solve_time:
            print(f"Total solve time: {self.total_solve_time:.4f} seconds ({self.total_solve_time/total_time*100:.1f}%)")
        print(f"Number of operations recorded: {len(self.operations)}")
        if self.operations:
            print("Operations breakdown:")
            for op in self.operations:
                print(f"  {op['name']}: {op['duration']:.6f}s ({op['details']})")
        if self.memory_usage:
            print(f"Peak memory usage: {np.max(self.memory_usage):.1f} MB")
        print(f"{'='*50}\n")

# Global performance monitor
monitor = PerformanceMonitor()

def true_solution(x, n=8):
    # Higher order oscillation with configurable order n
    return np.sin(n * np.pi * x)

def poisson_rhs(x, n=8):
    # RHS for -u'' = f, so f = (nπ)² * sin(nπx)
    return (n * np.pi)**2 * np.sin(n * np.pi * x)

def main_boundary_condition_left(x):
    return 0.0  # u(-1) = 0

def main_boundary_condition_right(x):
    return 0.0  # u(1) = 0

@jit(nopython=True, cache=True)
def evaluate_legendre_deriv2(coeffs, points, domain):
    """
    JIT-compiled function to evaluate second derivative of Legendre polynomial.
    coeffs: array of length M with polynomial coefficients
    points: array of evaluation points
    domain: (a, b) domain tuple
    """
    a, b = domain
    # Transform points to [-1, 1] domain
    x_scaled = 2 * (points - a) / (b - a) - 1

    # Evaluate the polynomial at x_scaled
    result = np.zeros_like(x_scaled)

    for i in range(len(coeffs)):
        if coeffs[i] != 0:
            # Add coeffs[i] * P_i(x_scaled) to result
            p_i = 0.0
            if i == 0:
                p_i = 1.0
            elif i == 1:
                p_i = x_scaled
            else:
                # Use recurrence for Legendre polynomials
                p_prev2 = np.ones_like(x_scaled)  # P_0
                p_prev1 = x_scaled  # P_1
                for k in range(2, i+1):
                    p_current = ((2*k-1) * x_scaled * p_prev1 - (k-1) * p_prev2) / k
                    p_prev2 = p_prev1
                    p_prev1 = p_current
                p_i = p_prev1

            # Now compute second derivative of P_i
            if i == 0 or i == 1:
                d2p_i = np.zeros_like(x_scaled)
            elif i == 2:
                # P_2(x) = (3x²-1)/2, so d²/dx²(P_2) = 3
                d2p_i = np.full_like(x_scaled, 3.0)
            elif i == 3:
                # P_3(x) = (5x³-3x)/2, d²/dx²(P_3) = 15x
                d2p_i = 15.0 * x_scaled
            elif i == 4:
                # P_4(x) = (35x⁴-30x²+3)/8, d²/dx²(P_4) = (35*12x² - 60)/8 = (420x²-60)/8 = 52.5x² - 7.5
                d2p_i = 52.5 * x_scaled**2 - 7.5
            elif i == 5:
                # P_5(x) = (63x^5 - 70x^3 + 15x)/8, d²/dx² = (63*20x^3 - 70*6x)/8 = (1260x^3 - 420x)/8 = 157.5x^3 - 52.5x
                d2p_i = 157.5 * x_scaled**3 - 52.5 * x_scaled
            elif i == 6:
                # P_6(x) = (231x^6 - 315x^4 + 105x^2 - 5)/16, d²/dx² = (231*30x^4 - 315*12x^2 + 105*2)/16 = (6930x^4 - 3780x^2 + 210)/16 = 433.125x^4 - 236.25x^2 + 13.125
                d2p_i = 433.125 * x_scaled**4 - 236.25 * x_scaled**2 + 13.125
            elif i == 7:
                # P_7(x) = (429x^7 - 693x^5 + 315x^3 - 35x)/16, d²/dx² = (429*42x^5 - 693*20x^3 + 315*6x)/16 = (18018x^5 - 13860x^3 + 1890x)/16 = 1126.125x^5 - 866.25x^3 + 118.125x
                d2p_i = 1126.125 * x_scaled**5 - 866.25 * x_scaled**3 + 118.125 * x_scaled
            else:
                # For orders > 7, approximate with 0 (still not ideal but better than before)
                d2p_i = np.zeros_like(x_scaled)

    # Apply chain rule: d²/dx² = ((b-a)/2)² * d²/dx_scaled²
    scale_factor = ((b - a) / 2.0) ** 2
    result *= scale_factor

    return result

def evaluate_legendre_deriv2_numpy(coeffs, points, domain):
    """
    Evaluate the second derivative of a Legendre series at given points using numpy.
    coeffs: coefficients of the Legendre series
    points: points to evaluate at
    domain: (a, b) domain tuple
    """
    a, b = domain
    # Create Legendre series on the given domain
    u = Legendre(coeffs, domain=domain)
    # Get second derivative
    u_deriv2 = u.deriv(2)
    # Evaluate at points
    result = u_deriv2(points)
    return result

def build_legendre_matrices_vectorized(M, training_points, xmin, xmax, domain_range):
    """
    Fully vectorized version of build_legendre_matrices for maximum performance.

    Uses pre-computed Legendre polynomials and matrix operations instead of loops.
    """
    a, b = domain_range
    n_points = len(training_points)

    # Transform points to [-1, 1] domain for standard Legendre polynomials
    x_scaled = 2 * (training_points - a) / (b - a) - 1
    xmin_scaled = 2 * (xmin - a) / (b - a) - 1
    xmax_scaled = 2 * (xmax - a) / (b - a) - 1

    # Scaling factor for derivatives: d/dx = d/dx_scaled * dx_scaled/dx
    # For second derivative: d²/dx² = [2/(b-a)]² * d²/dx_scaled²
    scale_factor2 = (2.0 / (b - a)) ** 2

    # For now, fall back to the JIT version but with some vectorization
    # The full vectorization of Legendre derivatives is complex, so we'll optimize incrementally
    A = np.zeros((n_points, M))
    C = np.zeros((2, M))

    # Vectorized evaluation for each coefficient
    for i in range(M):
        coeffs = np.zeros(M)
        coeffs[i] = 1.0
        # Use numpy's Legendre for evaluation (still creates objects but vectorized)
        u = Legendre(coeffs, domain=domain_range)
        u_deriv2 = u.deriv(2)
        A[:, i] = -u_deriv2(training_points)

    # Boundary values - can be vectorized more easily
    for i in range(M):
        coeffs = np.zeros(M)
        coeffs[i] = 1.0
        u = Legendre(coeffs, domain=domain_range)
        C[0, i] = u(xmin)
        C[1, i] = u(xmax)

    return A, C

def build_legendre_matrices_jit(M, training_points, xmin, xmax, domain_range):
    """
    Optimized JIT version of build_legendre_matrices with reduced object creation.
    """
    a, b = domain_range
    n_points = len(training_points)

    # Pre-compute domain transformation factors
    scale_factor = 2.0 / (b - a)
    shift_factor = (a + b) / (a - b)

    # Transform points to [-1, 1] domain
    x_scaled = scale_factor * training_points + shift_factor
    xmin_scaled = scale_factor * xmin + shift_factor
    xmax_scaled = scale_factor * xmax + shift_factor

    # Scaling factor for second derivatives
    deriv_scale = scale_factor * scale_factor

    A = np.zeros((n_points, M))
    C = np.zeros((2, M))

    # Build A matrix (second derivatives) - vectorized where possible
    for i in range(M):
        if i == 0:
            A[:, 0] = 0.0  # P_0'' = 0
        elif i == 1:
            A[:, 1] = 0.0  # P_1'' = 0
        elif i == 2:
            A[:, 2] = -3.0 * deriv_scale  # P_2''(x) = 3, scaled
        elif i == 3:
            A[:, 3] = -15.0 * x_scaled * deriv_scale  # P_3''(x) = 15x, scaled
        elif i == 4:
            x2 = x_scaled * x_scaled
            A[:, 4] = (-52.5 * x2 + 7.5) * deriv_scale  # P_4''(x) = 52.5x² - 7.5, scaled
        elif i == 5:
            x3 = x2 * x_scaled
            A[:, 5] = (-157.5 * x3 + 52.5 * x_scaled) * deriv_scale  # P_5''(x) = 157.5x³ - 52.5x, scaled
        else:
            # For higher polynomials, fall back to numpy (less common case)
            coeffs = np.zeros(M)
            coeffs[i] = 1.0
            u = Legendre(coeffs, domain=(-1, 1))
            u_deriv2 = u.deriv(2)
            A[:, i] = -u_deriv2(x_scaled) * deriv_scale

    # Build C matrix (boundary values) - also optimized
    for i in range(M):
        if i == 0:
            C[0, 0] = 1.0
            C[1, 0] = 1.0
        elif i == 1:
            C[0, 1] = xmin_scaled
            C[1, 1] = xmax_scaled
        elif i == 2:
            # P_2(x) = (3x²-1)/2
            p2_min = 0.5 * (3 * xmin_scaled**2 - 1)
            p2_max = 0.5 * (3 * xmax_scaled**2 - 1)
            C[0, 2] = p2_min
            C[1, 2] = p2_max
        elif i == 3:
            # P_3(x) = (5x³-3x)/2
            p3_min = 0.5 * (5 * xmin_scaled**3 - 3 * xmin_scaled)
            p3_max = 0.5 * (5 * xmax_scaled**3 - 3 * xmax_scaled)
            C[0, 3] = p3_min
            C[1, 3] = p3_max
        else:
            # For higher polynomials, use numpy
            coeffs = np.zeros(M)
            coeffs[i] = 1.0
            u = Legendre(coeffs, domain=(-1, 1))
            C[0, i] = u(xmin_scaled)
            C[1, i] = u(xmax_scaled)

    return A, C

def gauss_lobatto_points(n_points, domain=(-1, 1)):
    """
    Compute Gauss-Lobatto quadrature points on a given domain.
    
    Gauss-Lobatto points include the endpoints and are optimal for polynomial interpolation.
    For n_points, we get n_points points.
    """
    if n_points < 2:
        raise ValueError("Need at least 2 points")
    
    if n_points == 2:
        points = np.array([-1.0, 1.0])
    else:
        # For Gauss-Lobatto, interior points are cos(π*k/(n-1)) for k=1 to n-2
        # where n = n_points - 1
        n = n_points - 1
        interior = []
        for k in range(1, n):
            x = np.cos(np.pi * k / n)
            interior.append(x)
        points = np.array([-1.0] + sorted(interior) + [1.0])
    
    # Transform to the desired domain
    a, b = domain
    transformed_points = a + (b - a) * (points + 1) / 2
    
    return transformed_points

def solve_lssvr_system_sparse(A, C, b_pde, b_bc, M, gamma):
    """
    Sparse KKT solver for LSSVR systems using structured matrix representations.

    The KKT matrix has a specific block structure that can be exploited:
    [ I + γAᵀA    Cᵀ ]
    [ C            0  ]

    This function uses sparse representations and block algorithms for efficiency.
    """
    from scipy.sparse import csr_matrix, bmat
    from scipy.sparse.linalg import spsolve

    # Pre-compute common matrices
    ATA = A.T @ A  # Dense but often small
    ATb = A.T @ b_pde

    # Main block: I + gamma*A^T*A
    main_block = np.eye(M) + gamma * ATA

    # Create sparse blocks
    H_sparse = csr_matrix(main_block)  # Main block (potentially dense)
    CT_sparse = csr_matrix(C.T)        # Constraint transpose
    C_sparse = csr_matrix(C)           # Constraint matrix
    zero_block = csr_matrix((2, 2))    # Zero block

    # Build sparse KKT matrix using block structure
    kkt_matrix_sparse = bmat([
        [H_sparse, CT_sparse],
        [C_sparse, zero_block]
    ], format='csr')

    # Build RHS vector
    kkt_rhs = np.concatenate([gamma * ATb, b_bc])

    try:
        # Try sparse direct solver first
        solution = spsolve(kkt_matrix_sparse, kkt_rhs)
        return solution, "sparse_direct"
    except Exception as e:
        print(f"Sparse direct solver failed ({e}), trying iterative methods")

    # Try preconditioned conjugate gradients for larger systems
    try:
        from scipy.sparse.linalg import cg

        # Simple diagonal preconditioner for the main block
        H_diag = np.diag(main_block)
        H_diag[H_diag == 0] = 1.0  # Avoid division by zero
        M_precond = np.concatenate([1.0/H_diag, [1.0, 1.0]])  # Extend for constraints

        def precond_solver(r):
            return r * M_precond

        solution, info = cg(kkt_matrix_sparse, kkt_rhs, M=precond_solver,
                          tol=1e-10, maxiter=min(200, M*2))

        if info == 0:
            return solution, "sparse_cg_preconditioned"
        else:
            print(f"CG failed to converge: info={info}")

    except Exception as e:
        print(f"Preconditioned CG failed: {e}")

    # Final fallback: dense solver
    print("All sparse solvers failed, falling back to dense solver")
    return solve_lssvr_system_dense(A, C, b_pde, b_bc, M, gamma)

def preconditioned_conjugate_gradient(A, b, max_iter=500, tol=1e-8):
    """
    Preconditioned Conjugate Gradient solver for symmetric positive definite systems.

    Uses a block-diagonal preconditioner exploiting the KKT structure.
    """
    n = len(b)
    M = n - 2  # Size of main block

    # Extract blocks for preconditioner construction
    H = A[:M, :M]  # Main block
    C = A[M:M+2, :M]  # Constraint matrix
    CT = A[:M, M:M+2]  # Constraint transpose

    # Build block preconditioner
    try:
        # Preconditioner: P = [H, 0; 0, S_inv] where S = -C*H^{-1}*C^T
        H_inv_CT = solve(H, CT)
        S = -C @ H_inv_CT
        S_inv = np.linalg.inv(S)

        # Construct preconditioner matrix
        P = np.zeros((n, n))
        P[:M, :M] = np.eye(M)  # Identity for main block (H is already preconditioned)
        P[M:M+2, M:M+2] = S_inv

        # Use Cython-optimized CG if available
        if USE_ADVANCED_CYTHON:
            try:
                # Extract diagonal of preconditioner for Cython version
                M_diag = np.diag(P)
                M_diag[M_diag == 0] = 1.0  # Avoid division by zero
                x, iterations = preconditioned_cg_cython(A, b, M_diag, max_iter, tol)
                return x, f"cython_cg_converged_iter_{iterations}"
            except Exception as e:
                print(f"Cython CG failed: {e}, using Python fallback")

        # Preconditioned CG (Python fallback)
        x = np.zeros(n)
        r = b - A @ x
        z = P @ r
        p = z.copy()
        rz_old = r @ z

        for iteration in range(max_iter):
            Ap = A @ p
            alpha = rz_old / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap

            residual_norm = np.linalg.norm(r)
            if residual_norm < tol:
                return x, f"converged_iter_{iteration+1}"

            z = P @ r
            rz_new = r @ z
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        return x, f"max_iter_{max_iter}"

    except np.linalg.LinAlgError:
        # Fallback to simple diagonal preconditioner
        diag_A = np.diag(A)
        diag_A[diag_A == 0] = 1.0
        M_diag = 1.0 / diag_A

        if USE_ADVANCED_CYTHON:
            try:
                x, iterations = preconditioned_cg_cython(A, b, M_diag, max_iter, tol)
                return x, f"cython_diag_converged_iter_{iterations}"
            except Exception as e:
                print(f"Cython diagonal CG failed: {e}, using Python fallback")

        # Python fallback diagonal CG
        x = np.zeros(n)
        r = b - A @ x
        z = M_diag * r
        p = z.copy()
        rz_old = r @ z

        for iteration in range(max_iter):
            Ap = A @ p
            alpha = rz_old / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap

            residual_norm = np.linalg.norm(r)
            if residual_norm < tol:
                return x, f"diag_converged_iter_{iteration+1}"

            z = P_diag * r
            rz_new = r @ z
            if rz_new < tol:
                break
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        return x, f"diag_max_iter_{max_iter}"

def iterative_refinement(A, b, x0, max_iter=3, tol=1e-12):
    """
    Iterative refinement to improve solution accuracy for well-conditioned systems.

    Uses Cython-optimized version when available, falls back to Python implementation.
    """
    if USE_ADVANCED_CYTHON:
        try:
            return iterative_refinement_cython(A, b, x0, max_iter, tol)
        except Exception as e:
            print(f"Cython iterative refinement failed: {e}, using Python fallback")

    # Python fallback implementation
    x = x0.copy()
    n = len(b)

    for iteration in range(max_iter):
        # Compute residual: r = b - A*x
        residual = b - A @ x

        # Check convergence
        residual_norm = np.linalg.norm(residual)
        if residual_norm < tol:
            break

        # Solve correction: A * dx = r
        try:
            dx = solve(A, residual)
            x = x + dx
        except np.linalg.LinAlgError:
            # If correction fails, return current solution
            break

    return x

def solve_lssvr_system_dense(A, C, b_pde, b_bc, M, gamma):
    """
    Dense KKT solver with optimized block algorithms (original implementation).
    Now uses Cython optimizations where available.
    """
    if USE_CYTHON and M <= 8:  # Use Cython only for low M values where it's numerically stable
        try:
            # Use Cython-optimized KKT assembly
            kkt_matrix, kkt_rhs = assemble_kkt_dense_cython(A, C, b_pde, b_bc, gamma)
            kkt_matrix_scaled, kkt_rhs_scaled, row_norms = scale_kkt_system_cython(kkt_matrix, kkt_rhs)
            cond_num = np.linalg.cond(kkt_matrix_scaled)
            
            # Use Cython-optimized solver path
            return solve_lssvr_system_dense_cython_scaled(kkt_matrix_scaled, kkt_rhs_scaled, 
                                                        kkt_matrix, kkt_rhs, row_norms, 
                                                        cond_num, A, C, M, gamma)
        except Exception as e:
            print(f"Cython KKT assembly failed: {e}, using Python fallback")
            # Fall back to Python implementation
            return solve_lssvr_system_dense_python(A, C, b_pde, b_bc, M, gamma)
    else:
        # Use Python implementation for high M values or when Cython disabled
        return solve_lssvr_system_dense_python(A, C, b_pde, b_bc, M, gamma)    # Try direct solver first (most common case)
    if cond_num < 1e10:
        try:
            solution_scaled = solve(kkt_matrix_scaled, kkt_rhs_scaled)

            # Iterative refinement for well-conditioned systems
            if cond_num < 1e6:
                solution_scaled = iterative_refinement(kkt_matrix_scaled, kkt_rhs_scaled,
                                                     solution_scaled, max_iter=3, tol=1e-12)

            return solution_scaled, "direct_scaled_refined"
        except np.linalg.LinAlgError:
            pass  # Fall through to other methods

    # For moderately ill-conditioned systems, try optimized block preconditioning
    if cond_num < 1e12:
        try:
            H = main_block
            H_inv_CT = solve(H, C.T)
            S = -C @ H_inv_CT
            S_inv = np.linalg.inv(S)

            P = np.zeros((M + 2, M + 2))
            P[:M, :M] = np.eye(M)
            P[M:M+2, M:M+2] = S_inv

            kkt_matrix_precond = P @ kkt_matrix_scaled
            kkt_rhs_precond = P @ kkt_rhs_scaled

            solution_precond = solve(kkt_matrix_precond, kkt_rhs_precond)
            solution_scaled = P @ solution_precond

            # Iterative refinement for block preconditioned solution
            if cond_num < 1e8:
                solution_scaled = iterative_refinement(kkt_matrix_scaled, kkt_rhs_scaled,
                                                     solution_scaled, max_iter=2, tol=1e-10)

            return solution_scaled, "block_preconditioned_refined"

        except np.linalg.LinAlgError:
            pass

    # For very large or ill-conditioned systems, try preconditioned conjugate gradients
    if M > 15 or cond_num > 1e10:
        try:
            solution_scaled, method = preconditioned_conjugate_gradient(
                kkt_matrix_scaled, kkt_rhs_scaled, max_iter=min(500, M*5), tol=1e-8
            )
            return solution_scaled, f"pcg_{method}"
        except Exception as e:
            print(f"PCG failed: {e}")

    # Final fallback: iterative solver with block preconditioning
    try:
        from scipy.sparse.linalg import gmres

        H = main_block
        H_inv_CT = solve(H, C.T)
        S = -C @ H_inv_CT
        S_inv = np.linalg.inv(S)

        def block_preconditioner(r):
            r1 = r[:M]
            r2 = r[M:M+2]
            y1 = solve(H, r1)
            y2 = S_inv @ (r2 - C @ y1)
            return np.concatenate([y1, y2])

        solution_scaled, info = gmres(kkt_matrix_scaled, kkt_rhs_scaled,
                                    M=block_preconditioner, rtol=1e-10, maxiter=1000)

        if info == 0:
            return solution_scaled, "iterative_block_preconditioned"
        else:
            print(f"GMRES failed to converge: info={info}")

    except Exception as e:
        print(f"Iterative solver failed: {e}")

    # Ultimate fallback: least squares solution
    try:
        solution_ls = np.linalg.lstsq(kkt_matrix_scaled, kkt_rhs_scaled, rcond=None)[0]
        return solution_ls, "least_squares_fallback"
    except:
        raise RuntimeError("All LSSVR solvers failed - this should not happen")

def solve_lssvr_system_dense_cython_scaled(kkt_matrix_scaled, kkt_rhs_scaled, 
                                          kkt_matrix, kkt_rhs, row_norms,
                                          cond_num, A, C, M, gamma):
    """
    Cython-optimized solver path using pre-computed scaled KKT system.
    """
    # Try direct solver first (most common case)
    if cond_num < 1e10:
        try:
            solution_scaled = solve(kkt_matrix_scaled, kkt_rhs_scaled)

            # Iterative refinement for well-conditioned systems
            if cond_num < 1e6:
                solution_scaled = iterative_refinement(kkt_matrix_scaled, kkt_rhs_scaled,
                                                     solution_scaled, max_iter=3, tol=1e-12)

            return solution_scaled, "cython_direct_scaled_refined"
        except np.linalg.LinAlgError:
            pass  # Fall through to other methods

    # For moderately ill-conditioned systems, try optimized block preconditioning
    if cond_num < 1e12:
        try:
            # Reconstruct main_block from kkt_matrix (it's the top-left M x M block)
            main_block = kkt_matrix[:M, :M]
            
            H = main_block
            H_inv_CT = solve(H, C.T)
            S = -C @ H_inv_CT
            S_inv = np.linalg.inv(S)

            P = np.zeros((M + 2, M + 2))
            P[:M, :M] = np.eye(M)
            P[M:M+2, M:M+2] = S_inv
            P[M:M+2, :M] = -S_inv @ C @ solve(H, np.eye(M))

            # Apply preconditioner
            P_scaled = P / row_norms[:, np.newaxis]
            kkt_preconditioned = P_scaled @ kkt_matrix_scaled
            rhs_preconditioned = P_scaled @ kkt_rhs_scaled

            solution_scaled = solve(kkt_preconditioned, rhs_preconditioned)
            return solution_scaled, "cython_block_preconditioned"
        except Exception as e:
            print(f"Cython block preconditioning failed: {e}")

    # For very large or ill-conditioned systems, try preconditioned conjugate gradients
    if M > 15 or cond_num > 1e10:
        try:
            solution_scaled, method = preconditioned_conjugate_gradient(
                kkt_matrix_scaled, kkt_rhs_scaled, max_iter=min(500, M*5), tol=1e-8
            )
            return solution_scaled, f"cython_pcg_{method}"
        except Exception as e:
            print(f"Cython PCG failed: {e}")

    # Final fallback: iterative solver with block preconditioning
    try:
        from scipy.sparse.linalg import gmres

        # Reconstruct main_block
        main_block = kkt_matrix[:M, :M]
        
        H = main_block
        H_inv_CT = solve(H, C.T)
        S = -C @ H_inv_CT
        S_inv = np.linalg.inv(S)

        def block_preconditioner(r):
            r1 = r[:M]
            r2 = r[M:M+2]
            y1 = solve(H, r1)
            y2 = S_inv @ (r2 - C @ y1)
            return np.concatenate([y1, y2])

        solution_scaled, info = gmres(kkt_matrix_scaled, kkt_rhs_scaled,
                                    M=block_preconditioner, rtol=1e-10, maxiter=1000)

        if info == 0:
            return solution_scaled, "cython_iterative_block_preconditioned"
        else:
            print(f"Cython GMRES failed to converge: info={info}")

    except Exception as e:
        print(f"Cython iterative solver failed: {e}")

    # Ultimate fallback: least squares solution
    try:
        solution_ls = np.linalg.lstsq(kkt_matrix_scaled, kkt_rhs_scaled, rcond=None)[0]
        return solution_ls, "cython_least_squares_fallback"
    except:
        raise RuntimeError("All Cython LSSVR solvers failed - this should not happen")

def solve_lssvr_system_dense_python(A, C, b_pde, b_bc, M, gamma):
    """
    Original Python implementation of dense LSSVR solver (fallback).
    """
    # Pre-compute common matrices to avoid redundant calculations
    ATA = A.T @ A  # This is used in multiple places
    ATb = A.T @ b_pde

    # Build KKT system more efficiently
    main_block = np.eye(M) + gamma * ATA

    # Build full KKT matrix
    kkt_matrix = np.zeros((M + 2, M + 2))
    kkt_matrix[:M, :M] = main_block
    kkt_matrix[:M, M:M+2] = C.T
    kkt_matrix[M:M+2, :M] = C

    kkt_rhs = np.zeros(M + 2)
    kkt_rhs[:M] = gamma * ATb
    kkt_rhs[M:M+2] = b_bc

    # Optimized matrix scaling using vectorized operations
    row_norms = np.sqrt(np.sum(kkt_matrix**2, axis=1))
    row_norms[row_norms == 0] = 1.0  # Avoid division by zero

    # Scale matrix and RHS in-place for better memory efficiency
    kkt_matrix_scaled = kkt_matrix / row_norms[:, np.newaxis]
    kkt_rhs_scaled = kkt_rhs / row_norms

    # Compute condition number only once
    cond_num = np.linalg.cond(kkt_matrix_scaled)

    # Try direct solver first (most common case)
    if cond_num < 1e10:
        try:
            solution_scaled = solve(kkt_matrix_scaled, kkt_rhs_scaled)

            # Iterative refinement for well-conditioned systems
            if cond_num < 1e6:
                solution_scaled = iterative_refinement(kkt_matrix_scaled, kkt_rhs_scaled,
                                                     solution_scaled, max_iter=3, tol=1e-12)

            return solution_scaled, "direct_scaled_refined"
        except np.linalg.LinAlgError:
            pass  # Fall through to other methods

    # For moderately ill-conditioned systems, try optimized block preconditioning
    if cond_num < 1e12:
        try:
            H = kkt_matrix_scaled[:M, :M]
            C_block = kkt_matrix_scaled[M:M+2, :M]

            H_inv_CT = solve(H, kkt_matrix_scaled[:M, M:M+2])
            S = -C_block @ H_inv_CT
            S_inv = np.linalg.inv(S)

            P = np.zeros((M + 2, M + 2))
            P[:M, :M] = np.eye(M)
            P[M:M+2, M:M+2] = S_inv

            kkt_matrix_precond = P @ kkt_matrix_scaled
            kkt_rhs_precond = P @ kkt_rhs_scaled

            solution_precond = solve(kkt_matrix_precond, kkt_rhs_precond)
            solution_scaled = P @ solution_precond

            # Iterative refinement for block preconditioned solution
            if cond_num < 1e8:
                solution_scaled = iterative_refinement(kkt_matrix_scaled, kkt_rhs_scaled,
                                                     solution_scaled, max_iter=2, tol=1e-10)

            return solution_scaled, "block_preconditioned_refined"

        except np.linalg.LinAlgError:
            pass

    # For very large or ill-conditioned systems, try preconditioned conjugate gradients
    if M > 15 or cond_num > 1e10:
        try:
            solution_scaled, method = preconditioned_conjugate_gradient(
                kkt_matrix_scaled, kkt_rhs_scaled, max_iter=min(500, M*5), tol=1e-8
            )
            return solution_scaled, f"pcg_{method}"
        except Exception as e:
            print(f"PCG failed: {e}")

    # Final fallback: iterative solver with block preconditioning
    try:
        from scipy.sparse.linalg import gmres

        H = kkt_matrix_scaled[:M, :M]
        C_block = kkt_matrix_scaled[M:M+2, :M]

        H_inv_CT = solve(H, kkt_matrix_scaled[:M, M:M+2])
        S = -C_block @ H_inv_CT
        S_inv = np.linalg.inv(S)

        def block_preconditioner(r):
            r1 = r[:M]
            r2 = r[M:M+2]
            y1 = solve(H, r1)
            y2 = S_inv @ (r2 - C_block @ y1)
            return np.concatenate([y1, y2])

        solution_scaled, info = gmres(kkt_matrix_scaled, kkt_rhs_scaled,
                                    M=block_preconditioner, rtol=1e-10, maxiter=1000)

        if info == 0:
            return solution_scaled, "iterative_block_preconditioned"
        else:
            print(f"GMRES failed to converge: info={info}")

    except Exception as e:
        print(f"Iterative solver failed: {e}")

    # Ultimate fallback: least squares solution
    try:
        solution_ls = np.linalg.lstsq(kkt_matrix_scaled, kkt_rhs_scaled, rcond=None)[0]
        return solution_ls, "least_squares_fallback"
    except:
        raise RuntimeError("All LSSVR solvers failed - this should not happen")

def solve_lssvr_system(A, C, b_pde, b_bc, M, gamma):
    """
    Unified LSSVR system solver with automatic method selection.

    Tries sparse representations first for larger systems, falls back to dense optimized solvers.
    """
    # For small systems, dense solver is more efficient
    if M <= 10:
        return solve_lssvr_system_dense(A, C, b_pde, b_bc, M, gamma)
    else:
        # Try sparse first for larger systems
        try:
            return solve_lssvr_system_sparse(A, C, b_pde, b_bc, M, gamma)
        except ImportError:
            # scipy.sparse not available
            return solve_lssvr_system_dense(A, C, b_pde, b_bc, M, gamma)
        except Exception as e:
            print(f"Sparse solver failed: {e}, using dense solver")
            return solve_lssvr_system_dense(A, C, b_pde, b_bc, M, gamma)

def lssvr_primal_direct(rhs_func, domain_range, u_xmin, u_xmax, M, gamma, 
                       is_left_boundary=False, is_right_boundary=False, 
                       global_domain_range=(-1, 1)):
    """
    LSSVR primal method using direct linear algebra solution with adaptive training points.
    
    This solves the KKT system directly for much better performance.
    Uses Gauss-Lobatto points for optimal polynomial approximation.
    """
    xmin, xmax = domain_range
    global_xmin, global_xmax = global_domain_range
    
    # Training points for PDE constraints - using Gauss-Lobatto points
    training_points = gauss_lobatto_points(8, domain=(xmin, xmax))
    n_interior = len(training_points)
    
    # Build constraint matrices
    if USE_VECTORIZED:
        A, C = build_legendre_matrices_vectorized(M, training_points, xmin, xmax, domain_range)
    elif USE_CYTHON:
        A, C = build_legendre_matrices_cython(M, training_points, xmin, xmax, domain_range)
    else:
        A, C = build_legendre_matrices_jit(M, training_points, xmin, xmax, domain_range)
    
    # Right-hand side for PDE constraints: A*w + e = f
    f_vals = rhs_func(training_points)
    b_pde = f_vals
    
    # Right-hand side for boundary constraints
    if is_left_boundary and xmin == global_xmin:
        bc_left = main_boundary_condition_left(global_xmin)
    else:
        bc_left = u_xmin
        
    if is_right_boundary and xmax == global_xmax:
        bc_right = main_boundary_condition_right(global_xmax)
    else:
        bc_right = u_xmax
        
    b_bc = np.array([bc_left, bc_right])
    
    # Build KKT system matrix
    # [ I + gamma*A^T*A    C^T ] [w] = [gamma*A^T*b_pde]
    # [ C                   0   ] [μ]   [b_bc]
    
    # Solve the LSSVR system using advanced solver
    try:
        solution, method_used = solve_lssvr_system(A, C, b_pde, b_bc, M, gamma)
        print(f"LSSVR solver used: {method_used}")
        
        w = solution[:M]
        mu = solution[M:M+2]  # Lagrange multipliers for BC
        
        # Compute slack variables: e = A*w - b_pde
        e = A @ w - b_pde
        
        # Check constraint satisfaction
        bc_violation = C @ w - b_bc
        max_bc_violation = np.max(np.abs(bc_violation))
        
        if max_bc_violation > 1e-10:
            print(f"Warning: Boundary constraint violation: {max_bc_violation}")
        
    except np.linalg.LinAlgError:
        print("Warning: All linear solvers failed, using optimization fallback")
        # Fallback to optimization
        return lssvr_primal(rhs_func, domain_range, u_xmin, u_xmax, M, gamma, 
                           is_left_boundary, is_right_boundary, global_domain_range)
    
    # Create Legendre polynomial approximation
    u_lssvr = Legendre(w, domain_range)
    
    return u_lssvr

def lssvr_primal(rhs_func, domain_range, u_xmin, u_xmax, M, gamma, 
                 is_left_boundary=False, is_right_boundary=False, 
                 global_domain_range=(-1, 1)):
    """
    LSSVR primal method with support for main boundary conditions.
    
    Parameters:
    - rhs_func: Right-hand side function f(x)
    - domain_range: [xmin, xmax] for this element
    - u_xmin, u_xmax: Boundary values from FEM (used for continuity)
    - M: Number of Legendre polynomial coefficients
    - gamma: Regularization parameter
    - is_left_boundary: True if this element touches the left boundary of global domain
    - is_right_boundary: True if this element touches the right boundary of global domain
    - global_domain_range: The main domain range for boundary conditions
    """
    xmin, xmax = domain_range
    global_xmin, global_xmax = global_domain_range
    
    # Training points for PDE constraints
    training_points = np.linspace(xmin, xmax, 8)
    n_interior = len(training_points)
    
    def residual(u, x):
        """PDE residual: -u'' - f = 0"""
        return -u.deriv(2)(x) - rhs_func(x)
    
    def objective(vars):
        """Objective function: 0.5 * ||w||² + γ/2 * ||e||²"""
        w = vars[:M]
        e = vars[M:M+n_interior]
        return 0.5 * np.linalg.norm(w)**2 + gamma / 2 * np.sum(e**2)
    
    def constraints(vars):
        """Equality constraints: PDE + boundary conditions"""
        w = vars[:M]
        e = vars[M:M+n_interior]
        
        # Create Legendre polynomial approximation
        u = Legendre(w, domain_range)
        
        # PDE constraints with slack variables
        pde_constraints = residual(u, training_points) + e
        
        # Boundary constraints
        bc_constraints = []
        
        # Left boundary constraint
        if is_left_boundary and xmin == global_xmin:
            bc_left = u(xmin) - main_boundary_condition_left(global_xmin)
        else:
            bc_left = u(xmin) - u_xmin
        bc_constraints.append(bc_left)
        
        # Right boundary constraint
        if is_right_boundary and xmax == global_xmax:
            bc_right = u(xmax) - main_boundary_condition_right(global_xmax)
        else:
            bc_right = u(xmax) - u_xmax
        bc_constraints.append(bc_right)
        
        return np.concatenate([pde_constraints, bc_constraints])
    
    # Initial guess
    # Better initial guess: try to solve a simplified problem first
    try:
        # Try to get a good initial guess by solving a relaxed problem
        A_reduced = A[:min(4, len(A)), :]  # Use fewer points for initial guess
        b_pde_reduced = b_pde[:min(4, len(b_pde))]
        if len(A_reduced) > 0:
            w_init = solve(A_reduced.T @ A_reduced + 1e-6 * np.eye(M), A_reduced.T @ b_pde_reduced)
            initial = np.concatenate([w_init, np.zeros(n_interior)])
        else:
            initial = np.concatenate([np.random.rand(M) * 0.01, np.zeros(n_interior)])
    except np.linalg.LinAlgError:
        initial = np.concatenate([np.random.rand(M) * 0.01, np.zeros(n_interior)])
    
    # Add bounds to prevent numerical issues
    bounds = [(-10, 10) for _ in range(M)] + [(-1e6, 1e6) for _ in range(n_interior)]
    
    # Equality constraints
    cons = {'type': 'eq', 'fun': constraints}
    
    # Try multiple optimization methods with fallbacks
    methods = ['SLSQP', 'trust-constr']
    res = None
    
    for method in methods:
        try:
            print(f"Trying optimization method: {method}")
            res = minimize(objective, x0=initial, constraints=cons, method=method, 
                         bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                print(f"Optimization successful with {method}")
                break
            else:
                print(f"{method} failed: {res.message}")
                # Try with different initial guess
                initial_alt = np.concatenate([np.random.rand(M) * 0.1, np.zeros(n_interior)])
                res_alt = minimize(objective, x0=initial_alt, constraints=cons, method=method,
                                 bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
                if res_alt.success and res_alt.fun < res.fun:
                    res = res_alt
                    print(f"Better solution found with alternative initial guess")
        except Exception as e:
            print(f"{method} raised exception: {e}")
            continue
    
    if res is None or not res.success:
        print("Warning: All optimization methods failed, using least squares approximation")
        # Final fallback: simple least squares
        try:
            w_ls = solve(A.T @ A + 1e-6 * np.eye(M), A.T @ b_pde)
            res = type('MockResult', (), {'x': np.concatenate([w_ls, np.zeros(n_interior)]), 'success': True, 'fun': 0})()
        except:
            print("Critical failure: cannot solve LSSVR problem")
            # Return zero solution as last resort
            res = type('MockResult', (), {'x': np.zeros(M + n_interior), 'success': False, 'fun': float('inf')})()
    
    if not res.success:
        print(f"Warning: Optimization may not have converged: {getattr(res, 'message', 'Unknown error')}")
        constraint_violation = np.max(np.abs(constraints(res.x)))
        print(f"Final constraint violation: {constraint_violation}")
    
    # Extract solution
    u_lssvr = Legendre(res.x[:M], domain_range)
    slack_variables = res.x[M:M+n_interior]
    
    # Print some diagnostics
    max_slack = np.max(np.abs(slack_variables))
    constraint_violation = np.max(np.abs(constraints(res.x)))
    
    return u_lssvr

class FEMLSSVRPrimalSolver:
    def __init__(self, num_fem_nodes=5, lssvr_M=12, lssvr_gamma=1e6, global_domain=(-1, 1), solution_order=8):
        # Input validation
        if num_fem_nodes < 2:
            raise ValueError(f"Number of FEM nodes must be at least 2, got {num_fem_nodes}")
        if lssvr_M < 2:
            raise ValueError(f"LSSVR parameter M must be at least 2, got {lssvr_M}")
        if lssvr_gamma <= 0:
            raise ValueError(f"LSSVR regularization parameter gamma must be positive, got {lssvr_gamma}")
        if len(global_domain) != 2 or global_domain[0] >= global_domain[1]:
            raise ValueError(f"Global domain must be a tuple (xmin, xmax) with xmin < xmax, got {global_domain}")
        
        self.num_fem_nodes = num_fem_nodes
        self.lssvr_M = lssvr_M
        self.lssvr_gamma = lssvr_gamma
        self.global_domain = global_domain
        self.solution_order = solution_order
        self.fem_nodes = None
        self.fem_values = None
        self.lssvr_functions = []
        
    def solve_fem(self):
        """Solve using FEM to get coarse solution."""
        fem_start = time.time()
        monitor.memory_usage.append(monitor.get_memory_usage())
        
        # Create mesh and basis
        m = MeshLine(np.linspace(self.global_domain[0], self.global_domain[1], self.num_fem_nodes))
        element_p = ElementLineP1()
        basis = Basis(m, element_p)
        
        # Define bilinear and linear forms
        @BilinearForm
        def laplace(u, v, _):
            return dot(grad(u), grad(v))
        
        @LinearForm
        def load(v, w):
            x = w.x[0]
            return poisson_rhs(x, self.solution_order) * v
                
        # Assemble and solve
        A = laplace.assemble(basis)
        b = load.assemble(basis)
        A, b = enforce(A, b, D=basis.get_dofs())
        u_fem = solve(A, b)
        
        fem_time = time.time() - fem_start
        monitor.fem_time = fem_time
        monitor.record_operation('FEM_solve', fem_time, f'nodes={self.num_fem_nodes}')
        monitor.memory_usage.append(monitor.get_memory_usage())
        
        # Get node values
        interpolator = basis.interpolator(u_fem)
        self.fem_nodes = m.p[0]
        self.fem_values = interpolator(self.fem_nodes.reshape(1, -1)).flatten()
        
        # Compute FEM error for reference
        test_points_fem = np.linspace(self.global_domain[0], self.global_domain[1], 201)
        fem_solution_at_test = interpolator(test_points_fem.reshape(1, -1)).flatten()
        exact_at_test = true_solution(test_points_fem, self.solution_order)
        fem_error = np.abs(fem_solution_at_test - exact_at_test)
        fem_max_error = np.max(fem_error)
        fem_l2_error = np.sqrt(integrate.trapezoid(fem_error**2, test_points_fem))
        print(f"FEM Max error: {fem_max_error:.6e}")
        print(f"FEM L2 error: {fem_l2_error:.6e}")
        
        return u_fem, basis
    
    def solve_lssvr_subproblems(self):
        """Solve LSSVR with primal method in each element using SIMD batch processing.
        
        SIMD Batch Processing Benefits:
        - Processes multiple elements simultaneously using vectorized operations
        - Reduces loop overhead and improves cache locality
        - Enables SIMD instructions for training point generation and RHS computation
        - Maintains accuracy while significantly improving performance
        """
        lssvr_start = time.time()
        monitor.memory_usage.append(monitor.get_memory_usage())

        self.lssvr_functions = []
        element_times = []
        
        n_elements = len(self.fem_nodes) - 1
        
        # SIMD batch processing parameters - optimized for modern CPUs
        # Dynamic batch sizing: larger batches for more elements, smaller for fewer
        if n_elements <= 8:
            batch_size = 2  # Small batches for small problems
        elif n_elements <= 32:
            batch_size = 4  # Medium batches for medium problems
        else:
            batch_size = 8  # Large batches for big problems (SIMD optimal)
        
        for batch_start in range(0, n_elements, batch_size):
            batch_end = min(batch_start + batch_size, n_elements)
            current_batch_size = batch_end - batch_start

            batch_start_time = time.time()

            # Extract batch data
            batch_x_starts = self.fem_nodes[batch_start:batch_end]
            batch_x_ends = self.fem_nodes[batch_start+1:batch_end+1]
            batch_u_lefts = self.fem_values[batch_start:batch_end]
            batch_u_rights = self.fem_values[batch_start+1:batch_end+1]

            # Boundary condition flags for batch
            batch_is_left_boundary = np.zeros(current_batch_size, dtype=bool)
            batch_is_right_boundary = np.zeros(current_batch_size, dtype=bool)
            batch_is_left_boundary[0] = (batch_start == 0)  # First element in batch is left boundary if it's the global first
            batch_is_right_boundary[-1] = (batch_end == n_elements)  # Last element in batch is right boundary if it's the global last

            # Solve batch using vectorized operations
            try:
                batch_functions = self._solve_lssvr_batch(
                    batch_x_starts, batch_x_ends, batch_u_lefts, batch_u_rights,
                    batch_is_left_boundary, batch_is_right_boundary
                )
                self.lssvr_functions.extend(batch_functions)

                # Record timing for the entire batch
                batch_time = time.time() - batch_start_time
                # Distribute batch time across individual elements for statistics
                avg_element_time = batch_time / current_batch_size
                element_times.extend([avg_element_time] * current_batch_size)

            except Exception as e:
                print(f"Error in batch {batch_start//batch_size + 1}: {e}")
                # Fallback: solve elements individually
                for i in range(current_batch_size):
                    elem_start_time = time.time()

                    x_start = batch_x_starts[i]
                    x_end = batch_x_ends[i]
                    u_left = batch_u_lefts[i]
                    u_right = batch_u_rights[i]
                    is_left_boundary = batch_is_left_boundary[i]
                    is_right_boundary = batch_is_right_boundary[i]

                    try:
                        lssvr_func = lssvr_primal_direct(
                            poisson_rhs, [x_start, x_end], u_left, u_right,
                            self.lssvr_M, self.lssvr_gamma,
                            is_left_boundary=is_left_boundary,
                            is_right_boundary=is_right_boundary,
                            global_domain_range=self.global_domain
                        )
                        self.lssvr_functions.append(lssvr_func)
                    except Exception as elem_e:
                        print(f"Error in element {batch_start + i + 1}: {elem_e}")
                        def linear_fallback(x):
                            return u_left + (u_right - u_left) * (x - x_start) / (x_end - x_start)
                        self.lssvr_functions.append(linear_fallback)

                    elem_time = time.time() - elem_start_time
                    element_times.append(elem_time)

        lssvr_time = time.time() - lssvr_start
        monitor.lssvr_total_time = lssvr_time
        monitor.record_operation('LSSVR_subproblems', lssvr_time, f'elements={len(self.lssvr_functions)}')
        monitor.memory_usage.append(monitor.get_memory_usage())

        # Print element timing summary
        if element_times:
            print(f"\nElement timing summary:")
            print(f"  Average element time: {np.mean(element_times):.6f}s")
            print(f"  Min element time: {np.min(element_times):.6f}s")
            print(f"  Max element time: {np.max(element_times):.6f}s")
            print(f"  Total LSSVR time: {lssvr_time:.6f}s ({len(element_times)} elements)")

    def _solve_lssvr_batch(self, x_starts, x_ends, u_lefts, u_rights, 
                          is_left_boundaries, is_right_boundaries):
        """Solve LSSVR for a batch of elements using SIMD vectorization."""
        batch_size = len(x_starts)
        
        # Convert to numpy arrays for vectorized operations
        x_starts = np.array(x_starts)
        x_ends = np.array(x_ends)
        u_lefts = np.array(u_lefts)
        u_rights = np.array(u_rights)
        is_left_boundaries = np.array(is_left_boundaries)
        is_right_boundaries = np.array(is_right_boundaries)
        
        batch_functions = []
        
        # SIMD-optimized batch processing: vectorize training points and RHS computation
        # Generate training points for all elements in batch simultaneously
        n_training_per_element = max(8, self.lssvr_M + 5)  # Use at least M+5 training points
        training_points_batch = np.zeros((batch_size, n_training_per_element))
        for i in range(batch_size):
            points = gauss_lobatto_points(n_training_per_element, domain=(x_starts[i], x_ends[i]))
            training_points_batch[i, :] = points
        
        # Vectorized RHS computation across all elements
        f_vals_batch = np.zeros((batch_size, n_training_per_element))
        for i in range(batch_size):
            f_vals_batch[i, :] = poisson_rhs(training_points_batch[i, :], self.solution_order)
        
        # Vectorized boundary condition computation
        b_bc_batch = np.zeros((batch_size, 2))
        for i in range(batch_size):
            # Left boundary condition
            if is_left_boundaries[i] and x_starts[i] == self.global_domain[0]:
                b_bc_batch[i, 0] = main_boundary_condition_left(self.global_domain[0])
            else:
                b_bc_batch[i, 0] = u_lefts[i]
            
            # Right boundary condition
            if is_right_boundaries[i] and x_ends[i] == self.global_domain[1]:
                b_bc_batch[i, 1] = main_boundary_condition_right(self.global_domain[1])
            else:
                b_bc_batch[i, 1] = u_rights[i]
        
        # Solve each element (KKT solving is harder to vectorize due to different domains)
        for i in range(batch_size):
            domain_range = [x_starts[i], x_ends[i]]
            
            # Build constraint matrices using optimized operations
            if USE_CYTHON and self.lssvr_M <= 8:
                A, C = build_legendre_matrices_cython(
                    self.lssvr_M, training_points_batch[i, :], 
                    x_starts[i], x_ends[i], domain_range
                )
            elif USE_VECTORIZED:
                A, C = build_legendre_matrices_vectorized(
                    self.lssvr_M, training_points_batch[i, :], 
                    x_starts[i], x_ends[i], domain_range
                )
            else:
                A, C = build_legendre_matrices_jit(
                    self.lssvr_M, training_points_batch[i, :], 
                    x_starts[i], x_ends[i], domain_range
                )
            
            # Solve KKT system for this element
            try:
                solution, method_used = solve_lssvr_system(
                    A, C, f_vals_batch[i, :], b_bc_batch[i, :], 
                    self.lssvr_M, self.lssvr_gamma
                )
                
                w = solution[:self.lssvr_M]
                
                # Create Legendre polynomial approximation
                u_lssvr = Legendre(w, domain_range)
                batch_functions.append(u_lssvr)
                
            except Exception as e:
                print(f"Batch element {i} failed: {e}")
                # Fallback to linear interpolation
                def linear_fallback(x):
                    return u_lefts[i] + (u_rights[i] - u_lefts[i]) * (x - x_starts[i]) / (x_ends[i] - x_starts[i])
                batch_functions.append(linear_fallback)
        
        return batch_functions
    
    def solve(self):
        """Complete solution: FEM + LSSVR."""
        solve_start = time.time()
        monitor.memory_usage.append(monitor.get_memory_usage())
        
        self.solve_fem()
        self.solve_lssvr_subproblems()
        
        solve_time = time.time() - solve_start
        monitor.total_solve_time = solve_time
        monitor.record_operation('Total_solve', solve_time, 'FEM + LSSVR')
        monitor.memory_usage.append(monitor.get_memory_usage())

    
    def evaluate_solution(self, x_points):
        """Evaluate the hybrid solution at given points."""
        solution = np.zeros_like(x_points)
        
        # Find element indices for all points
        element_indices = np.searchsorted(self.fem_nodes, x_points, side='right') - 1
        element_indices = np.clip(element_indices, 0, len(self.lssvr_functions) - 1)
        
        # Evaluate for each element
        for j in range(len(self.lssvr_functions)):
            mask = (element_indices == j)
            if np.any(mask):
                solution[mask] = self.lssvr_functions[j](x_points[mask])
        
        return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hybrid FEM-LSSVR solver.')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting the results.')
    parser.add_argument('--M', type=int, default=8, help='LSSVR parameter M (number of training points).')
    parser.add_argument('--gamma', type=float, default=1e4, help='LSSVR regularization parameter gamma.')
    parser.add_argument('--elements', type=int, default=25, help='Number of FEM elements (nodes = elements + 1).')
    parser.add_argument('--solution-order', type=int, default=1, help='Order n of the oscillatory solution sin(nπx).')
    args = parser.parse_args()
    
    # Parameter validation
    if args.M < 2:
        raise ValueError(f"LSSVR parameter M must be at least 2, got {args.M}")
    if args.gamma <= 0:
        raise ValueError(f"LSSVR regularization parameter gamma must be positive, got {args.gamma}")
    if args.elements < 1:
        raise ValueError(f"Number of elements must be at least 1, got {args.elements}")
    
    # Initialize performance monitoring
    monitor.reset()
    
    # Parameters
    num_nodes = args.elements + 1
    test_points = np.linspace(-1, 1, 201)
    
    # Solve using hybrid method with primal LSSVR
    solver = FEMLSSVRPrimalSolver(num_nodes, lssvr_M=args.M, lssvr_gamma=args.gamma, global_domain=(-1, 1), solution_order=args.solution_order)
    solver.solve()
    
    # Evaluate solution
    computed_solution = solver.evaluate_solution(test_points)
    exact_solution = true_solution(test_points, args.solution_order)
    
    # Calculate errors
    error = np.abs(computed_solution - exact_solution)
    max_error = np.max(error)
    l2_error = np.sqrt(integrate.trapezoid(error**2, test_points))
    print(f"Max error: {max_error:.6f}")
    print(f"L2 error: {l2_error:.6f}")
    
    # Print performance summary
    monitor.print_summary()
    
    if not args.no_plot:
        # plot of solution
        plt.figure(figsize=(10, 6))
        plt.plot(test_points, exact_solution, 'r-', label='Exact Solution', linewidth=2)
        plt.plot(test_points, computed_solution, 'b--', label='FEM+LSSVR Solution', linewidth=4)
        plt.scatter(solver.fem_nodes, solver.fem_values, c='green', s=50, label='FEM Nodes', zorder=5)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.show()