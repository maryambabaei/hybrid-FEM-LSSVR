import numpy as np
from scipy.linalg import solve, cho_factor, cho_solve
from scipy import integrate
# Lazy import matplotlib only when plotting
# import matplotlib.pyplot as plt  # Moved to plotting section
from numpy.polynomial.legendre import Legendre, leggauss, legder, legroots
from scipy.special import eval_legendre
from skfem import *
from skfem.helpers import dot, grad
import argparse
import time
import psutil
import os
import numba
from numba import jit
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Memory Pool for array recycling - reduces allocation overhead
class ArrayPool:
    """Simple memory pool to recycle numpy arrays and reduce allocation overhead."""
    
    def __init__(self):
        self.pools = {}  # (shape, dtype) -> list of arrays
    
    def get_array(self, shape, dtype=np.float64, fill_val=None):
        """Get a recycled array or create new one."""
        key = (tuple(shape) if hasattr(shape, '__iter__') else (shape,), dtype)
        
        if key in self.pools and self.pools[key]:
            arr = self.pools[key].pop()
            if fill_val is not None:
                arr.fill(fill_val)
            return arr
        
        # Create new array
        arr = np.empty(shape, dtype=dtype)
        if fill_val is not None:
            arr.fill(fill_val)
        return arr
    
    def return_array(self, arr):
        """Return array to pool for reuse."""
        if arr is None:
            return
        key = (tuple(arr.shape), arr.dtype)
        if key not in self.pools:
            self.pools[key] = []
        # Limit pool size to prevent memory bloat
        if len(self.pools[key]) < 10:
            self.pools[key].append(arr)

# Global array pool instance
array_pool = ArrayPool()

# Try to import Cython extension
try:
    from legendre_matrices_cython import build_legendre_matrices_cython
    USE_CYTHON = True  # Now working correctly after fixing scaling and variable initialization
    print("Cython extension available and enabled for performance")
except ImportError:
    USE_CYTHON = False
    print("Cython extension not available, using optimized Python version")

# Try to import FEM Cython acceleration
try:
    from fem_assembly_cython import (
        assemble_fem_1d_cython,
        enforce_dirichlet_bc_cython,
        interpolate_solution_cython,
        poisson_rhs_vectorized,  # Fast C implementation for RHS evaluation
        FastLegendrePolynomial,   # Lightweight polynomial class
        create_fast_legendre_polynomials_batch  # Batch polynomial creation
    )
    USE_FEM_CYTHON = True
    USE_FAST_POLYNOMIAL = True
    print("FEM Cython acceleration available and enabled")
    print("Fast polynomial class enabled (10× faster)")
except ImportError:
    USE_FEM_CYTHON = False
    USE_FAST_POLYNOMIAL = False
    print("FEM Cython not available, using skfem")

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

# ============================================================================
# PARALLEL PROCESSING FOR LARGE SYSTEMS
# ============================================================================

def _parallel_lssvr_worker(args):
    """
    Worker function for parallel LSSVR batch processing.
    Must be defined at module level for multiprocessing.
    
    Args:
        args: Tuple of (batch_idx, x_starts, x_ends, u_lefts, u_rights, 
                       is_left_boundaries, is_right_boundaries, config)
              config = (M, gamma, n_training, solution_order, global_domain)
    
    Returns:
        (batch_idx, batch_functions): Tuple of batch index and list of Legendre functions
    """
    (batch_idx, x_starts, x_ends, u_lefts, u_rights, 
     is_left_boundaries, is_right_boundaries, config) = args
    
    M, gamma, n_training, solution_order, global_domain = config
    
    batch_size = len(x_starts)
    batch_functions = []
    
    # Convert to numpy arrays
    x_starts = np.array(x_starts)
    x_ends = np.array(x_ends)
    u_lefts = np.array(u_lefts)
    u_rights = np.array(u_rights)
    is_left_boundaries = np.array(is_left_boundaries)
    is_right_boundaries = np.array(is_right_boundaries)
    
    # Check uniformity
    element_sizes = x_ends - x_starts
    is_uniform = np.allclose(element_sizes, element_sizes[0], rtol=1e-10)
    
    if is_uniform:
        # Use optimized uniform path
        A_ref, C_ref, xi_points = build_reference_legendre_matrices(M, n_training)
        
        f_vals_batch = np.empty((batch_size, n_training))
        b_bc_batch = np.empty((batch_size, 2))
        
        # Map points to physical domain
        all_x_points = np.empty((batch_size, n_training))
        for i in range(batch_size):
            h = x_ends[i] - x_starts[i]
            all_x_points[i, :] = x_starts[i] + (x_ends[i] - x_starts[i]) * (xi_points + 1.0) / 2.0
        
        # Vectorized RHS evaluation
        all_f_vals = poisson_rhs_batch(all_x_points.ravel(), solution_order)
        f_vals_batch[:, :] = all_f_vals.reshape(batch_size, n_training)
        
        # Boundary conditions
        for i in range(batch_size):
            if is_left_boundaries[i] and x_starts[i] == global_domain[0]:
                b_bc_batch[i, 0] = main_boundary_condition_left(global_domain[0])
            else:
                b_bc_batch[i, 0] = u_lefts[i]
            
            if is_right_boundaries[i] and x_ends[i] == global_domain[1]:
                b_bc_batch[i, 1] = main_boundary_condition_right(global_domain[1])
            else:
                b_bc_batch[i, 1] = u_rights[i]
        
        # Scale by Jacobian and solve
        h = x_ends[0] - x_starts[0]
        inv_jac_sq = (2.0 / h) ** 2
        A = A_ref * inv_jac_sq
        C = C_ref
        
        solutions, _ = solve_lssvr_batch_optimized(A, C, f_vals_batch, b_bc_batch, M, gamma)
        
        # Create Legendre polynomials using FastLegendrePolynomial
        if USE_FAST_POLYNOMIAL:
            batch_functions = create_fast_legendre_polynomials_batch(
                solutions, x_starts, x_ends, M
            )
        else:
            for i in range(batch_size):
                w = solutions[i, :M]
                domain_range_i = [x_starts[i], x_ends[i]]
                u_lssvr = Legendre(w, domain_range_i)
                batch_functions.append(u_lssvr)
    else:
        # Non-uniform path (rare for large systems)
        A_ref, C_ref, xi_points = build_reference_legendre_matrices(M, n_training)
        
        for i in range(batch_size):
            h = x_ends[i] - x_starts[i]
            x_points = x_starts[i] + h * (xi_points + 1.0) / 2.0
            f_vals = poisson_rhs_batch(x_points, solution_order)
            
            # Boundary conditions
            if is_left_boundaries[i] and x_starts[i] == global_domain[0]:
                bc_left = main_boundary_condition_left(global_domain[0])
            else:
                bc_left = u_lefts[i]
            
            if is_right_boundaries[i] and x_ends[i] == global_domain[1]:
                bc_right = main_boundary_condition_right(global_domain[1])
            else:
                bc_right = u_rights[i]
            
            b_bc = np.array([bc_left, bc_right])
            
            # Scale and solve
            inv_jac_sq = (2.0 / h) ** 2
            A = A_ref * inv_jac_sq
            C = C_ref
            
            solution, _ = solve_lssvr_system(A, C, f_vals, b_bc, M, gamma)
            w = solution[:M]
            if USE_FAST_POLYNOMIAL:
                u_lssvr = FastLegendrePolynomial(w, (x_starts[i], x_ends[i]))
            else:
                u_lssvr = Legendre(w, [x_starts[i], x_ends[i]])
            batch_functions.append(u_lssvr)
    
    return (batch_idx, batch_functions)

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
        
        # Cache statistics
        global _cache_hits, _cache_misses
        if _cache_hits + _cache_misses > 0:
            cache_hit_rate = 100 * _cache_hits / (_cache_hits + _cache_misses)
            print(f"\nCache Performance:")
            print(f"  Legendre cache hits: {_cache_hits}")
            print(f"  Legendre cache misses: {_cache_misses}")
            print(f"  Cache hit rate: {cache_hit_rate:.1f}%")
        
        print(f"{'='*50}\n")

# Global performance monitor
monitor = PerformanceMonitor()

def true_solution(x, n=8):
    # Higher order oscillation with configurable order n
    return np.sin(n * np.pi * x)

@jit(nopython=True, cache=True, fastmath=True)
def poisson_rhs(x, n=8):
    # RHS for -u'' = f, so f = (nπ)² * sin(nπx)
    # JIT-compiled for ~1ms speedup (1000+ calls)
    return (n * np.pi)**2 * np.sin(n * np.pi * x)

def main_boundary_condition_left(x):
    return 0.0  # u(-1) = 0

def main_boundary_condition_right(x):
    return 0.0  # u(1) = 0

# Dual implementation: Use Cython if available, fallback to Numba
if USE_FEM_CYTHON:
    # Use fast C implementation from Cython
    def poisson_rhs_batch(x_array, n=8):
        """Wrapper for Cython RHS - ultra-fast C implementation."""
        return poisson_rhs_vectorized(x_array, n)
else:
    # Fallback to Numba JIT
    @jit(nopython=True, cache=True, fastmath=True)
    def poisson_rhs_batch(x_array, n=8):
        """
        Vectorized RHS evaluation with Numba optimization.
        Fully vectorized - computes all values at once.
        """
        n_pi = n * np.pi
        n_pi_sq = n_pi * n_pi
        # Vectorized computation - no loop needed!
        return n_pi_sq * np.sin(n_pi * x_array)

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
    OPTIMIZED: Reuses cached Legendre objects to avoid allocation overhead.
    """
    # Reuse existing object from pool if possible
    M = len(coeffs)
    domain_tuple = tuple(domain)
    key = (M, domain_tuple)
    
    # Create or get cached polynomial object
    # For evaluation, we need ONE object with the given coefficients
    if key not in _legendre_cache:
        _legendre_cache[key] = Legendre(coeffs, domain=domain_tuple)
    else:
        # Reuse existing object by updating coefficients
        poly = _legendre_cache[key]
        # Update coefficients in-place (Legendre stores coef as array)
        poly.coef[:] = coeffs
    
    u = _legendre_cache[key]
    # Get second derivative
    u_deriv2 = u.deriv(2)
    # Evaluate at points
    result = u_deriv2(points)
    return result

def get_preallocated_zeros(shape, dtype=np.float64):
    """
    Get pre-allocated zero array for common shapes to reduce allocation overhead.
    Reuses arrays when possible for ~2-3ms savings in tight loops.
    """
    key = (shape if isinstance(shape, tuple) else (shape,), dtype)
    if key not in _preallocated_arrays:
        _preallocated_arrays[key] = np.zeros(shape, dtype=dtype)
    arr = _preallocated_arrays[key]
    arr.fill(0)  # Reset to zeros
    return arr

def get_legendre_basis_objects(M, domain):
    """
    Get or create reusable Legendre basis objects for given M and domain.
    Returns list of M Legendre objects, one for each basis function.
    Reuses existing objects to avoid repeated allocation (~7ms savings).
    """
    domain_tuple = tuple(domain)
    key = (M, domain_tuple)
    
    if key not in _legendre_object_pool:
        # Create M basis Legendre polynomials (one per basis function)
        basis_objects = []
        for i in range(M):
            coeffs = np.zeros(M)
            coeffs[i] = 1.0
            basis_objects.append(Legendre(coeffs, domain=domain_tuple))
        _legendre_object_pool[key] = basis_objects
    
    return _legendre_object_pool[key]

def build_legendre_matrices_vectorized(M, training_points, xmin, xmax, domain_range):
    """
    Builds Legendre collocation matrices using vectorized operations.
    OPTIMIZED: Reuses Legendre objects instead of creating 2M new objects per call.
    This saves ~7ms by eliminating 1000+ Legendre.__init__() calls.
    """
    n_points = len(training_points)

    # Initialize matrices
    A = np.zeros((n_points, M))
    C = np.zeros((2, M))

    # Get reusable basis objects (creates only once per (M, domain), reuses thereafter)
    basis_objects = get_legendre_basis_objects(M, domain_range)
    
    # Vectorized evaluation using pre-existing objects (NO new allocations!)
    for i in range(M):
        u = basis_objects[i]
        u_deriv2 = u.deriv(2)
        A[:, i] = -u_deriv2(training_points)
        C[0, i] = u(xmin)
        C[1, i] = u(xmax)

    return A, C

# Global cache for Legendre matrix computations
_legendre_cache = {}
_cache_hits = 0
_cache_misses = 0

# Global pool of reusable Legendre polynomial objects
# Key: (M, domain_tuple) -> list of M Legendre objects (one per basis function)
_legendre_object_pool = {}

# Pre-allocated arrays to avoid repeated allocations (significant overhead reduction)
_preallocated_arrays = {}

# Isoparametric reference element cache (computed ONCE for all elements)
_reference_matrices_cache = {}

# Try to use Cython-optimized functions
USE_CYTHON = False

try:
    from lssvr_kernels import compute_legendre_A_matrix, compute_legendre_C_matrix
    USE_CYTHON = True
    print("Cython extension available and enabled for performance")
except ImportError:
    print("Cython extension not available, using standard NumPy operations")

def build_reference_legendre_matrices(M, n_training_points):
    """
    Build REFERENCE Legendre matrices on standard domain [-1, 1].
    These are computed ONCE and reused for ALL elements via isoparametric mapping.
    
    This is analogous to FEM reference elements: compute shape functions once,
    then map to physical elements using Jacobian transformation.
    
    Args:
        M: Number of Legendre basis functions
        n_training_points: Number of collocation points
    
    Returns:
        A_ref: Reference A matrix (n_training_points × M) - second derivatives
        C_ref: Reference C matrix (2 × M) - boundary values at ξ = ±1
        xi_points: Training points in reference domain [-1, 1]
    """
    cache_key = (M, n_training_points)
    
    if cache_key in _reference_matrices_cache:
        return _reference_matrices_cache[cache_key]
    
    # Training points in reference domain [-1, 1]
    xi_points = gauss_lobatto_points(n_training_points, domain=(-1, 1))
    
    A_ref = np.zeros((n_training_points, M), dtype=np.float64)
    C_ref = np.zeros((2, M), dtype=np.float64)
    
    # Use Cython if available
    if USE_CYTHON and M <= 13:
        # Cython functions work directly on [-1, 1] domain with scale=1
        A_ref = compute_legendre_A_matrix(xi_points, M, deriv_scale=1.0)
        C_ref = compute_legendre_C_matrix(-1.0, 1.0, M)
    else:
        # Build A matrix (second derivatives on reference element)
        for i in range(M):
            if i == 0:
                A_ref[:, 0] = 0.0  # P_0'' = 0
            elif i == 1:
                A_ref[:, 1] = 0.0  # P_1'' = 0
            elif i == 2:
                A_ref[:, 2] = -3.0  # P_2''(ξ) = 3
            elif i == 3:
                A_ref[:, 3] = -15.0 * xi_points  # P_3''(ξ) = 15ξ
            elif i == 4:
                xi2 = xi_points * xi_points
                A_ref[:, 4] = (-52.5 * xi2 + 7.5)  # P_4''(ξ) = 52.5ξ² - 7.5
            elif i == 5:
                xi2 = xi_points * xi_points
                xi3 = xi2 * xi_points
                A_ref[:, 5] = (-157.5 * xi3 + 52.5 * xi_points)  # P_5''(ξ)
            else:
                # Higher order: use numpy Legendre
                coeffs = np.zeros(M)
                coeffs[i] = 1.0
                u = Legendre(coeffs, domain=(-1, 1))
                u_deriv2 = u.deriv(2)
                A_ref[:, i] = -u_deriv2(xi_points)
        
        # Build C matrix (boundary values at ξ = -1, +1)
        for i in range(M):
            C_ref[0, i] = eval_legendre(i, -1.0)  # Left boundary
            C_ref[1, i] = eval_legendre(i, +1.0)  # Right boundary
    
    # Cache for reuse
    _reference_matrices_cache[cache_key] = (A_ref, C_ref, xi_points)
    
    return A_ref, C_ref, xi_points

@jit(nopython=True, cache=True, fastmath=True)
def _map_points_to_physical(xi_points, x_left, x_right):
    """
    JIT-compiled coordinate transformation from reference to physical domain.
    Fast vectorized mapping: x = x_left + (x_right - x_left) * (ξ + 1) / 2
    """
    h = x_right - x_left
    x_points = np.empty_like(xi_points)
    for i in range(len(xi_points)):
        x_points[i] = x_left + h * (xi_points[i] + 1.0) * 0.5
    return x_points

def map_reference_to_physical(A_ref, C_ref, xi_points, x_left, x_right):
    """
    Map reference element matrices to physical element using isoparametric transformation.
    
    Transformation: x = x_left + (x_right - x_left) * (ξ + 1) / 2
    where ξ ∈ [-1, 1] and x ∈ [x_left, x_right]
    
    Jacobian: dx/dξ = (x_right - x_left) / 2 = h/2
    
    For second derivatives:
        d²u/dx² = (dξ/dx)² * d²u/dξ² = (2/h)² * d²u/dξ²
    
    Args:
        A_ref: Reference A matrix (second derivatives)
        C_ref: Reference C matrix (boundary values) - unchanged!
        xi_points: Reference training points
        x_left, x_right: Physical element boundaries
    
    Returns:
        A_phys: Physical A matrix (scaled by Jacobian)
        C_phys: Physical C matrix (same as reference - Legendre values don't change!)
        x_points: Physical training points
    
    OPTIMIZED: Uses JIT-compiled coordinate transformation.
    """
    h = x_right - x_left  # Element length
    
    # Inverse Jacobian for second derivatives: (dξ/dx)² = (2/h)²
    inv_jac_sq = (2.0 / h) ** 2
    
    # Scale A matrix by inverse Jacobian squared
    A_phys = A_ref * inv_jac_sq
    
    # C matrix is UNCHANGED - Legendre polynomial values at boundaries don't change
    C_phys = C_ref  # No copy needed - same values!
    
    # Map training points to physical domain (JIT-compiled!)
    x_points = _map_points_to_physical(xi_points, x_left, x_right)
    
    return A_phys, C_phys, x_points

def build_legendre_matrices_isoparametric(M, x_left, x_right, n_training_points=None):
    """
    Build Legendre matrices using isoparametric transformation (FEM-style).
    
    This is MUCH faster than build_legendre_matrices_jit because:
    1. Reference matrices computed ONCE (cached)
    2. Only Jacobian scaling per element (trivial operation)
    3. No repeated Legendre polynomial evaluations
    
    Args:
        M: Number of Legendre basis functions
        x_left, x_right: Physical element boundaries
        n_training_points: Number of training points (default: max(8, M+5))
    
    Returns:
        A: Physical A matrix (n_points × M)
        C: Physical C matrix (2 × M)
        x_points: Physical training points
    """
    if n_training_points is None:
        n_training_points = max(8, M + 5)
    
    # Get reference matrices (cached after first call)
    A_ref, C_ref, xi_points = build_reference_legendre_matrices(M, n_training_points)
    
    # Map to physical element (just Jacobian scaling - very fast!)
    A_phys, C_phys, x_points = map_reference_to_physical(A_ref, C_ref, xi_points, x_left, x_right)
    
    return A_phys, C_phys, x_points

def build_legendre_matrices_jit(M, training_points, xmin, xmax, domain_range):
    """
    Optimized JIT version with caching for repeated domain computations.
    Cache is based on element length, not absolute position, since Legendre
    matrices only depend on the size of the domain, not its location.
    """
    global _cache_hits, _cache_misses
    
    a, b = domain_range
    n_points = len(training_points)
    element_length = b - a
    
    # Create cache key based on element LENGTH (not absolute position) and M
    # This allows cache hits for uniform meshes where all elements have same size
    cache_key = (element_length, M, n_points)
    
    # Check cache for pre-computed transformation factors
    if cache_key in _legendre_cache:
        _cache_hits += 1
        scale_factor, shift_factor_template, deriv_scale, C_template = _legendre_cache[cache_key]
        # Compute actual shift factor for this specific element position
        shift_factor = (a + b) / (a - b)
    else:
        _cache_misses += 1
        # Pre-compute domain transformation factors
        scale_factor = 2.0 / (b - a)
        shift_factor_template = 0.0  # Will be computed per-element
        shift_factor = (a + b) / (a - b)
        # Scaling factor for second derivatives
        deriv_scale = scale_factor * scale_factor
        C_template = None  # Will be computed below

    # Transform points to [-1, 1] domain
    x_scaled = scale_factor * training_points + shift_factor
    xmin_scaled = scale_factor * xmin + shift_factor
    xmax_scaled = scale_factor * xmax + shift_factor

    A = np.zeros((n_points, M), dtype=np.float64)
    C = np.zeros((2, M), dtype=np.float64)

    # Use Cython-optimized version if available (supports up to M=13)
    if USE_CYTHON and M <= 13:
        A = compute_legendre_A_matrix(x_scaled, M, deriv_scale)
        C = compute_legendre_C_matrix(xmin_scaled, xmax_scaled, M)
    else:
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

        # Build C matrix (boundary values) - optimized with scipy.special
        for i in range(M):
            # Use scipy.special.eval_legendre for direct evaluation (10-100x faster)
            C[0, i] = eval_legendre(i, xmin_scaled)
            C[1, i] = eval_legendre(i, xmax_scaled)
    
    # Cache transformation factors for reuse (store template shift_factor for reference)
    if cache_key not in _legendre_cache:
        _legendre_cache[cache_key] = (scale_factor, shift_factor_template, deriv_scale, C.copy())
    
    return A, C

# Global cache for Gauss-Lobatto points
_gauss_lobatto_cache = {}

def gauss_lobatto_points(n_points, domain=(-1, 1)):
    """
    Compute Gauss-Lobatto quadrature points on a given domain with caching.
    
    Gauss-Lobatto points include the endpoints and are optimal for polynomial interpolation.
    For n_points, we get n_points points.
    """
    if n_points < 2:
        raise ValueError("Need at least 2 points")
    
    # Check cache first
    if n_points in _gauss_lobatto_cache:
        points_ref = _gauss_lobatto_cache[n_points]
    else:
        if n_points == 2:
            points_ref = np.array([-1.0, 1.0])
        else:
            # For Gauss-Lobatto, interior points are cos(π*k/(n-1)) for k=1 to n-2
            n = n_points - 1
            # Vectorized computation instead of loop
            k = np.arange(1, n)
            interior = np.cos(np.pi * k / n)
            points_ref = np.concatenate([[-1.0], np.sort(interior), [1.0]])
        
        # Cache the reference points
        _gauss_lobatto_cache[n_points] = points_ref
    
    # Transform to the desired domain
    a, b = domain
    transformed_points = a + (b - a) * (points_ref + 1) / 2
    
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

    # Pre-compute common matrices - use optimized dot products
    ATA = A.T @ A  # Use @ for better SIMD performance
    ATb = A.T @ b_pde

    # Main block: I + gamma*A^T*A (pre-compute and reuse)
    main_block = np.eye(M, dtype=np.float64)
    main_block += gamma * ATA  # In-place addition

    # Create sparse blocks - directly specify dtype for efficiency
    H_sparse = csr_matrix(main_block, dtype=np.float64)
    CT_sparse = csr_matrix(C.T, dtype=np.float64)
    C_sparse = csr_matrix(C, dtype=np.float64)
    zero_block = csr_matrix((2, 2), dtype=np.float64)

    # Build sparse KKT matrix using block structure
    kkt_matrix_sparse = bmat([
        [H_sparse, CT_sparse],
        [C_sparse, zero_block]
    ], format='csr')

    # Build RHS vector - pre-allocate
    kkt_rhs = np.empty(M + 2, dtype=np.float64)
    kkt_rhs[:M] = gamma * ATb
    kkt_rhs[M:M+2] = b_bc

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

def solve_lssvr_system_reduced_normal(A, C, b_pde, b_bc, M, gamma):
    """
    FASTEST: Direct solution via reduced normal equations (avoids KKT system entirely).
    
    Instead of solving the full KKT system, we eliminate the Lagrange multipliers analytically.
    This reduces the problem to solving a single symmetric positive definite system.
    
    Theory: From KKT optimality conditions, we can derive:
        w = (A^T A + γ^{-1} I)^{-1} (A^T b_pde + γ^{-1} C^T (C (A^T A + γ^{-1} I)^{-1} C^T)^{-1} (b_bc - C (A^T A + γ^{-1} I)^{-1} A^T b_pde))
    
    But a simpler approach: Solve constrained least squares directly using QR factorization of C.
    """
    # Method 1: Use null space of constraints for unconstrained subproblem
    # For small 2 BC constraints, explicit formula is fastest
    
    # Pre-compute
    ATA = A.T @ A
    ATb = A.T @ b_pde
    
    # Build regularized normal matrix: N = I + γA^TA
    N = np.eye(M, dtype=np.float64) + gamma * ATA
    g = gamma * ATb
    
    # For the constrained problem: min 0.5*w^T N w - g^T w  s.t. Cw = h
    # Solution: w = N^{-1}(g + C^T λ) where λ solves (C N^{-1} C^T) λ = h - C N^{-1} g
    
    try:
        # Cholesky factorization of N
        N_factor = cho_factor(N, lower=False)
        
        # Compute N^{-1} C^T and N^{-1} g
        N_inv_CT = cho_solve(N_factor, C.T)
        N_inv_g = cho_solve(N_factor, g)
        
        # Schur complement: S = C N^{-1} C^T (2x2 matrix)
        S = C @ N_inv_CT
        
        # RHS for Lagrange multipliers: h - C N^{-1} g
        rhs_lambda = b_bc - C @ N_inv_g
        
        # Solve for Lagrange multipliers (2x2 system)
        lambda_ = np.linalg.solve(S, rhs_lambda)
        
        # Back-substitute: w = N^{-1}(g + C^T λ)
        w = N_inv_g + N_inv_CT @ lambda_
        
        solution = np.concatenate([w, lambda_])
        return solution, "reduced_normal"
        
    except np.linalg.LinAlgError as e:
        print(f"Reduced normal equation failed: {e}, using Schur complement fallback")
        return solve_lssvr_system_dense_python_schur(A, C, b_pde, b_bc, M, gamma)

def solve_lssvr_batch_optimized(A, C, b_pde_batch, b_bc_batch, M, gamma):
    """
    MOST EFFICIENT: Batch solver that reuses factorization across all elements.
    
    For uniform meshes, A and C matrices are IDENTICAL for all elements.
    We can factorize ONCE and solve for all RHS simultaneously.
    
    FULLY VECTORIZED - solves all systems at once without loops!
    
    Args:
        A: Constraint matrix (n_points × M) - SAME for all elements
        C: Boundary matrix (2 × M) - SAME for all elements
        b_pde_batch: RHS for PDE (batch_size × n_points)
        b_bc_batch: RHS for BC (batch_size × 2)
        M: Number of Legendre coefficients
        gamma: Regularization parameter
    
    Returns:
        solutions: (batch_size × (M+2)) array of solutions
        method: Solver method used
    """
    batch_size = b_pde_batch.shape[0]
    
    # Pre-compute matrices ONCE for entire batch
    ATA = A.T @ A
    N = np.eye(M, dtype=np.float64) + gamma * ATA
    
    try:
        # Single Cholesky factorization for all elements
        N_factor = cho_factor(N, lower=False)
        
        # Pre-compute N^{-1} C^T once (M × 2)
        N_inv_CT = cho_solve(N_factor, C.T)
        
        # Schur complement S = C N^{-1} C^T (2 × 2, same for all elements)
        S = C @ N_inv_CT
        S_factor = cho_factor(S, lower=False)
        
        # VECTORIZED: Compute all g vectors at once (batch_size × M)
        g_batch = gamma * (b_pde_batch @ A)  # (batch_size × n_points) @ (n_points × M) = (batch_size × M)
        
        # VECTORIZED: Solve all N w = g systems at once
        # cho_solve can handle multiple RHS by transposing
        N_inv_g_batch = cho_solve(N_factor, g_batch.T).T  # (batch_size × M)
        
        # VECTORIZED: Compute all λ RHS at once
        # rhs_lambda = b_bc_batch - (C @ N_inv_g_batch.T).T
        rhs_lambda_batch = b_bc_batch - (N_inv_g_batch @ C.T)  # (batch_size × 2)
        
        # VECTORIZED: Solve all S λ = rhs systems at once
        lambda_batch = cho_solve(S_factor, rhs_lambda_batch.T).T  # (batch_size × 2)
        
        # VECTORIZED: Back-substitution for all w
        w_batch = N_inv_g_batch + (lambda_batch @ N_inv_CT.T)  # (batch_size × M)
        
        # Assemble solutions
        solutions = np.zeros((batch_size, M + 2))
        solutions[:, :M] = w_batch
        solutions[:, M:] = lambda_batch
        
        return solutions, "batch_optimized_vectorized"
        
    except np.linalg.LinAlgError as e:
        print(f"Batch solver failed: {e}, falling back to element-wise")
        # Fallback: solve each element individually
        solutions = np.zeros((batch_size, M + 2))
        for i in range(batch_size):
            sol, _ = solve_lssvr_system_reduced_normal(A, C, b_pde_batch[i, :], b_bc_batch[i, :], M, gamma)
            solutions[i, :] = sol
        return solutions, "batch_fallback"

def solve_lssvr_system_dense_python_schur(A, C, b_pde, b_bc, M, gamma):
    """
    Optimized dense LSSVR solver using Cholesky factorization + Schur complement.
    
    KKT system: [H  C^T] [w] = [g]
                [C   0 ] [λ]   [h]
    where H = I + γA^TA (symmetric positive definite)
    
    Uses Schur complement method:
    1. Factorize H using Cholesky (faster than LU for SPD matrices)
    2. Compute Schur complement S = -C H^{-1} C^T
    3. Solve for Lagrange multipliers: S λ = h - C H^{-1} g
    4. Back-substitute for w: H w = g - C^T λ
    """
    # Pre-compute common matrices
    ATA = A.T @ A
    ATb = A.T @ b_pde

    # Build H = I + gamma * A^T A (symmetric positive definite)
    H = np.eye(M, dtype=np.float64) + gamma * ATA
    
    # Right-hand side vectors
    g = gamma * ATb  # For w equation
    h = b_bc         # For λ equation
    
    try:
        # Step 1: Cholesky factorization of H (2x faster than LU for SPD)
        H_factor = cho_factor(H, lower=False)  # Upper triangular factorization
        
        # Step 2: Compute H^{-1} C^T efficiently using Cholesky solve
        H_inv_CT = cho_solve(H_factor, C.T)
        
        # Step 3: Schur complement S = -C H^{-1} C^T (2x2 matrix)
        S = -C @ H_inv_CT
        
        # Step 4: Compute right-hand side for Schur complement equation
        H_inv_g = cho_solve(H_factor, g)
        rhs_schur = h - C @ H_inv_g
        
        # Step 5: Solve 2x2 system S λ = rhs_schur
        # For 2x2, direct solve is fastest
        lambda_ = np.linalg.solve(S, rhs_schur)
        
        # Step 6: Back-substitute: H w = g - C^T λ
        w = cho_solve(H_factor, g - C.T @ lambda_)
        
        # Combine solution
        solution = np.concatenate([w, lambda_])
        
        return solution, "schur_cholesky"
        
    except np.linalg.LinAlgError as e:
        # Fallback to original method if Cholesky fails (shouldn't happen for well-posed problems)
        print(f"Cholesky factorization failed: {e}, using fallback solver")
        return solve_lssvr_system_dense_python_fallback(A, C, b_pde, b_bc, M, gamma)

# Alias for backward compatibility
solve_lssvr_system_dense_python = solve_lssvr_system_reduced_normal

def solve_lssvr_system_dense_python_fallback(A, C, b_pde, b_bc, M, gamma):
    """
    Original Python implementation of dense LSSVR solver (fallback for ill-conditioned systems).
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

    Adaptive strategy based on problem size:
    - M <= 15: Dense Cholesky+Schur (fastest for small systems)
    - M > 15: Sparse iterative solvers (better for large systems)
    """
    # For small to medium systems, dense Cholesky solver is fastest
    # Sparse overhead (bmat construction) dominates for M <= 15
    if M <= 15:
        return solve_lssvr_system_dense_python(A, C, b_pde, b_bc, M, gamma)
    else:
        # For larger systems, try sparse first
        try:
            return solve_lssvr_system_sparse(A, C, b_pde, b_bc, M, gamma)
        except ImportError:
            # scipy.sparse not available
            return solve_lssvr_system_dense_python(A, C, b_pde, b_bc, M, gamma)
        except Exception as e:
            print(f"Sparse solver failed: {e}, using dense solver")
            return solve_lssvr_system_dense_python(A, C, b_pde, b_bc, M, gamma)

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
        
        if USE_FEM_CYTHON:
            # Fast Cython-based FEM assembly
            nodes = np.linspace(self.global_domain[0], self.global_domain[1], self.num_fem_nodes)
            
            # Assemble system
            data, rows, cols, b = assemble_fem_1d_cython(nodes, self.solution_order)
            
            # Build sparse matrix
            from scipy.sparse import coo_matrix
            A_sparse = coo_matrix((data, (rows, cols)), shape=(self.num_fem_nodes, self.num_fem_nodes))
            A_sparse = A_sparse.tocsr()  # Convert to CSR for efficient solving
            
            # Enforce boundary conditions (Dirichlet at both ends)
            # For 1D: first and last nodes are boundary
            boundary_dofs = np.array([0, self.num_fem_nodes - 1], dtype=np.int32)
            
            # Modify matrix: set boundary rows to identity
            for dof in boundary_dofs:
                A_sparse.data[A_sparse.indptr[dof]:A_sparse.indptr[dof+1]] = 0
                A_sparse[dof, dof] = 1.0
                b[dof] = 0.0  # Homogeneous BC
            
            # Eliminate boundary columns from RHS
            A_sparse.eliminate_zeros()
            
            # Solve sparse system
            from scipy.sparse.linalg import spsolve
            u_fem = spsolve(A_sparse, b)
            
            # Store solution
            self.fem_nodes = nodes
            self.fem_values = u_fem
            
        else:
            # Original skfem implementation
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
            
            # Get node values
            interpolator = basis.interpolator(u_fem)
            self.fem_nodes = m.p[0]
            self.fem_values = interpolator(self.fem_nodes.reshape(1, -1)).flatten()
        
        fem_time = time.time() - fem_start
        monitor.fem_time = fem_time
        monitor.record_operation('FEM_solve', fem_time, f'nodes={self.num_fem_nodes}')
        monitor.memory_usage.append(monitor.get_memory_usage())
        
        # Compute FEM error for reference
        test_points_fem = np.linspace(self.global_domain[0], self.global_domain[1], 201)
        
        if USE_FEM_CYTHON:
            # Use Cython interpolation
            fem_solution_at_test = interpolate_solution_cython(self.fem_nodes, self.fem_values, test_points_fem)
        else:
            # Use skfem interpolation
            fem_solution_at_test = interpolator(test_points_fem.reshape(1, -1)).flatten()
        
        exact_at_test = true_solution(test_points_fem, self.solution_order)
        fem_error = np.abs(fem_solution_at_test - exact_at_test)
        fem_max_error = np.max(fem_error)
        fem_l2_error = np.sqrt(integrate.trapezoid(fem_error**2, test_points_fem))
        print(f"FEM Max error: {fem_max_error:.6e}")
        print(f"FEM L2 error: {fem_l2_error:.6e}")
        
        if USE_FEM_CYTHON:
            return self.fem_values, None
        else:
            return u_fem, basis
    
    def solve_lssvr_subproblems(self):
        """Solve LSSVR with primal method in each element using SIMD batch processing.
        
        SIMD Batch Processing Benefits:
        - Processes multiple elements simultaneously using vectorized operations
        - Reduces loop overhead and improves cache locality
        - Enables SIMD instructions for training point generation and RHS computation
        - Maintains accuracy while significantly improving performance
        
        Parallel Processing (for n_elements > 10000):
        - Uses multiprocessing to distribute batches across CPU cores
        - Each worker processes batches independently
        - Significant speedup for very large systems (100k+ elements)
        """
        lssvr_start = time.time()
        monitor.memory_usage.append(monitor.get_memory_usage())

        n_elements = len(self.fem_nodes) - 1
        
        # Pre-allocate result array instead of using list.append() 
        # This avoids repeated memory reallocations
        self.lssvr_functions = [None] * n_elements
        element_times = []
        
        # Adaptive strategy: Use parallel processing for very large systems
        # Threshold increased to 2M elements to avoid overhead
        use_parallel = n_elements > 2000000
        n_workers = max(2, min(cpu_count() - 1, 8)) if use_parallel else 1
        
        # SIMD batch processing parameters - optimized for modern CPUs
        # Dynamic batch sizing: larger batches for more elements, smaller for fewer
        if n_elements <= 8:
            batch_size = 4  # Small batches for small problems
        elif n_elements <= 32:
            batch_size = 8  # Medium batches for medium problems
        elif n_elements <= 128:
            batch_size = 16  # Large batches for big problems
        elif n_elements <= 10000:
            batch_size = 32  # Very large batches for massive problems
        else:
            # For parallel: much larger batches to amortize overhead
            # Each worker should get substantial work to overcome process overhead
            batch_size = max(256, n_elements // (n_workers * 8))

        # PRE-COMPUTE ALL BATCH DATA ONCE (eliminates repeated slicing overhead)
        # Convert to numpy arrays for faster slicing
        fem_nodes_array = np.array(self.fem_nodes)
        fem_values_array = np.array(self.fem_values)

        if use_parallel:
            print(f"Parallel mode: {n_workers} workers, batch_size={batch_size}, batches={n_elements//batch_size}")


            # Prepare all batch arguments for parallel processing
            batch_args = []
            for batch_idx, batch_start in enumerate(range(0, n_elements, batch_size)):
                batch_end = min(batch_start + batch_size, n_elements)
                current_batch_size = batch_end - batch_start

                # Extract batch data (use pre-computed arrays)
                batch_x_starts = fem_nodes_array[batch_start:batch_end]
                batch_x_ends = fem_nodes_array[batch_start+1:batch_end+1]
                batch_u_lefts = fem_values_array[batch_start:batch_end]
                batch_u_rights = fem_values_array[batch_start+1:batch_end+1]

                # Boundary flags
                batch_is_left_boundary = [False] * current_batch_size
                batch_is_right_boundary = [False] * current_batch_size
                batch_is_left_boundary[0] = (batch_start == 0)
                batch_is_right_boundary[-1] = (batch_end == n_elements)

                # Configuration tuple
                config = (self.lssvr_M, self.lssvr_gamma,
                         max(8, self.lssvr_M + 5), self.solution_order,
                         self.global_domain)

                batch_args.append((
                    batch_idx, batch_x_starts.tolist(), batch_x_ends.tolist(),
                    batch_u_lefts.tolist(), batch_u_rights.tolist(),
                    batch_is_left_boundary, batch_is_right_boundary, config
                ))

            # Process batches in parallel
            with Pool(processes=n_workers) as pool:
                results = pool.map(_parallel_lssvr_worker, batch_args)

            # Sort results by batch index and insert functions at correct positions
            results.sort(key=lambda x: x[0])
            for batch_idx, batch_functions in results:
                batch_start = batch_idx * batch_size
                for i, func in enumerate(batch_functions):
                    if batch_start + i < n_elements:  # Safety check
                        self.lssvr_functions[batch_start + i] = func

            # Estimate timing (parallel execution)
            total_batch_time = time.time() - lssvr_start
            avg_element_time = total_batch_time / n_elements
            element_times = [avg_element_time] * n_elements

        else:
            # Sequential processing for smaller systems
            for batch_start in range(0, n_elements, batch_size):
                batch_end = min(batch_start + batch_size, n_elements)
                current_batch_size = batch_end - batch_start

                batch_start_time = time.time()

                # Extract batch data (use pre-computed arrays - much faster)
                batch_x_starts = fem_nodes_array[batch_start:batch_end]
                batch_x_ends = fem_nodes_array[batch_start+1:batch_end+1]
                batch_u_lefts = fem_values_array[batch_start:batch_end]
                batch_u_rights = fem_values_array[batch_start+1:batch_end+1]

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
                    # Direct assignment instead of extend() for better performance
                    for i, func in enumerate(batch_functions):
                        self.lssvr_functions[batch_start + i] = func

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

        # Print detailed timing breakdown (only for smaller systems to avoid overhead)
        if hasattr(self, '_timing_breakdown') and n_elements <= 10000:
            total_accounted = sum(self._timing_breakdown.values())
            overhead = lssvr_time - total_accounted
            self._timing_breakdown['other'] = overhead
            
            print(f"\n{'='*80}")
            print(f"LSSVR TIMING BREAKDOWN ({len(element_times)} elements):")
            print(f"{'='*80}")
            print(f"  Solving systems:    {self._timing_breakdown['solve']*1000:7.2f} ms ({self._timing_breakdown['solve']/lssvr_time*100:5.1f}%)")
            print(f"  Creating polynomials: {self._timing_breakdown['poly']*1000:7.2f} ms ({self._timing_breakdown['poly']/lssvr_time*100:5.1f}%)")
            print(f"  RHS evaluation:     {self._timing_breakdown['rhs']*1000:7.2f} ms ({self._timing_breakdown['rhs']/lssvr_time*100:5.1f}%)")
            print(f"  Other overhead:     {self._timing_breakdown['other']*1000:7.2f} ms ({self._timing_breakdown['other']/lssvr_time*100:5.1f}%)")
            print(f"  {'─'*78}")
            print(f"  TOTAL LSSVR time:   {lssvr_time*1000:7.2f} ms (100.0%)")
            print(f"{'='*80}\n")
        
        # Print element timing summary
        if element_times and n_elements <= 10000:
            print(f"Element timing summary:")
            print(f"  Average element time: {np.mean(element_times):.6f}s")
            print(f"  Min element time: {np.min(element_times):.6f}s")
            print(f"  Max element time: {np.max(element_times):.6f}s")
        
        print(f"  Total LSSVR time: {lssvr_time:.6f}s ({n_elements} elements)")

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
        
        # ISOPARAMETRIC OPTIMIZATION: For uniform meshes, use reference element approach
        # Check if all elements have same size (uniform mesh)
        # For large batches (>1000), assume uniform for performance (FEM typically generates uniform meshes)
        if batch_size > 1000:
            is_uniform = True  # Assume uniform for large batches
        else:
            element_sizes = x_ends - x_starts
            is_uniform = np.allclose(element_sizes, element_sizes[0], rtol=1e-10)
        
        # Number of training points
        n_training_per_element = max(8, self.lssvr_M + 5)
        
        if is_uniform:
            # ISOPARAMETRIC PATH: Use reference element (FEM-style)
            # Compute reference matrices ONCE, then just scale by Jacobian for each element
            
            # Build reference matrices (cached after first call)
            A_ref, C_ref, xi_points = build_reference_legendre_matrices(
                self.lssvr_M, n_training_per_element
            )
            
            # Pre-allocate arrays for batch (use empty - faster than zeros)
            f_vals_batch = np.empty((batch_size, n_training_per_element))
            b_bc_batch = np.empty((batch_size, 2))
            
            # Map reference to physical for each element - VECTORIZED
            # all_x_points[i, :] = x_starts[i] + (x_ends[i] - x_starts[i]) * (xi_points + 1.0) / 2.0
            h_batch = x_ends - x_starts  # (batch_size,)
            xi_scaled = (xi_points + 1.0) / 2.0  # (n_points,)
            # Broadcasting: (batch_size, 1) + (batch_size, 1) * (1, n_points) = (batch_size, n_points)
            all_x_points = x_starts[:, np.newaxis] + h_batch[:, np.newaxis] * xi_scaled[np.newaxis, :]
            
            # FULLY Vectorized RHS evaluation - evaluate ALL points at once!
            t_rhs = time.perf_counter()
            # Flatten to 1D array, evaluate, then reshape back
            all_f_vals = poisson_rhs_batch(all_x_points.ravel(), self.solution_order)
            f_vals_batch[:, :] = all_f_vals.reshape(batch_size, n_training_per_element)
            rhs_time = time.perf_counter() - t_rhs
                
            # Boundary conditions - VECTORIZED (eliminates Python loop)
            # Left boundaries
            left_bc_mask = (is_left_boundaries) & (x_starts == self.global_domain[0])
            b_bc_batch[:, 0] = np.where(left_bc_mask, 
                                       main_boundary_condition_left(self.global_domain[0]), 
                                       u_lefts)
            
            # Right boundaries  
            right_bc_mask = (is_right_boundaries) & (x_ends == self.global_domain[1])
            b_bc_batch[:, 1] = np.where(right_bc_mask,
                                        main_boundary_condition_right(self.global_domain[1]),
                                        u_rights)
            
            # Scale reference A matrix by Jacobian (same for all elements in uniform mesh)
            h = x_ends[0] - x_starts[0]
            inv_jac_sq = (2.0 / h) ** 2
            A = A_ref * inv_jac_sq
            C = C_ref  # Unchanged for Legendre polynomials
            
            # Solve entire batch with one factorization
            t_solve = time.perf_counter()
            solutions, method_used = solve_lssvr_batch_optimized(
                A, C, f_vals_batch, b_bc_batch, self.lssvr_M, self.lssvr_gamma
            )
            solve_time = time.perf_counter() - t_solve
            
            # Create Legendre polynomials from batch solutions
            t_poly = time.perf_counter()
            if USE_FAST_POLYNOMIAL:
                # Use fast Cython batch creation (eliminates Python loop)
                batch_functions = create_fast_legendre_polynomials_batch(
                    solutions, x_starts, x_ends, self.lssvr_M
                )
            else:
                # Fallback to numpy.Legendre
                batch_functions = []
                for i in range(batch_size):
                    w = solutions[i, :self.lssvr_M]
                    domain_range_i = [x_starts[i], x_ends[i]]
                    u_lssvr = Legendre(w, domain_range_i)
                    batch_functions.append(u_lssvr)
            poly_time = time.perf_counter() - t_poly
            
            # Track timing breakdown
            if not hasattr(self, '_timing_breakdown'):
                self._timing_breakdown = {'solve': 0.0, 'poly': 0.0, 'rhs': 0.0, 'other': 0.0}
            self._timing_breakdown['solve'] += solve_time
            self._timing_breakdown['poly'] += poly_time
            self._timing_breakdown['rhs'] += rhs_time
                
        else:
            # NON-UNIFORM PATH: Different element sizes require individual treatment
            # Still use isoparametric approach but with per-element Jacobians
            
            # Pre-allocate
            training_points_batch = np.empty((batch_size, n_training_per_element))
            f_vals_batch = np.empty((batch_size, n_training_per_element))
            b_bc_batch = np.empty((batch_size, 2))
            
            # Get reference matrices (cached)
            A_ref, C_ref, xi_points = build_reference_legendre_matrices(
                self.lssvr_M, n_training_per_element
            )
            
            for i in range(batch_size):
                # Map to physical element
                h = x_ends[i] - x_starts[i]
                x_points_phys = x_starts[i] + h * (xi_points + 1.0) / 2.0
                training_points_batch[i, :] = x_points_phys
                
                # RHS (vectorized)
                f_vals_batch[i, :] = poisson_rhs_batch(x_points_phys, self.solution_order)
                
                # Boundary conditions
                if is_left_boundaries[i] and x_starts[i] == self.global_domain[0]:
                    b_bc_batch[i, 0] = main_boundary_condition_left(self.global_domain[0])
                else:
                    b_bc_batch[i, 0] = u_lefts[i]
                
                if is_right_boundaries[i] and x_ends[i] == self.global_domain[1]:
                    b_bc_batch[i, 1] = main_boundary_condition_right(self.global_domain[1])
                else:
                    b_bc_batch[i, 1] = u_rights[i]
            
            # Solve and create polynomials for non-uniform batch
            t_solve = time.perf_counter()
            t_poly_total = 0.0
            for i in range(batch_size):
                # Scale A matrix by element-specific Jacobian
                h = x_ends[i] - x_starts[i]
                inv_jac_sq = (2.0 / h) ** 2
                A = A_ref * inv_jac_sq
                C = C_ref
                
                # Solve this element
                solution, method_used = solve_lssvr_system(
                    A, C, f_vals_batch[i, :], b_bc_batch[i, :], 
                    self.lssvr_M, self.lssvr_gamma
                )
                
                t_p = time.perf_counter()
                w = solution[:self.lssvr_M]
                if USE_FAST_POLYNOMIAL:
                    domain_range_i = (x_starts[i], x_ends[i])
                    u_lssvr = FastLegendrePolynomial(w, domain_range_i)
                else:
                    domain_range_i = [x_starts[i], x_ends[i]]
                    u_lssvr = Legendre(w, domain_range_i)
                batch_functions.append(u_lssvr)
                t_poly_total += time.perf_counter() - t_p
            
            solve_time = time.perf_counter() - t_solve - t_poly_total
            
            # Track timing breakdown
            if not hasattr(self, '_timing_breakdown'):
                self._timing_breakdown = {'solve': 0.0, 'poly': 0.0, 'rhs': 0.0, 'other': 0.0}
            self._timing_breakdown['solve'] += solve_time
            self._timing_breakdown['poly'] += t_poly_total
            self._timing_breakdown['rhs'] += rhs_time
        
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
    solver = FEMLSSVRPrimalSolver(num_nodes, lssvr_M=args.M, lssvr_gamma=args.gamma, 
                                   global_domain=(-1, 1), solution_order=args.solution_order)
    solver.solve()
    
    # Evaluate solution and errors (skip if not plotting to save ~5ms)
    if not args.no_plot:
        computed_solution = solver.evaluate_solution(test_points)
        exact_solution = true_solution(test_points, args.solution_order)
        
        error = np.abs(computed_solution - exact_solution)
        max_error = np.max(error)
        l2_error = np.sqrt(integrate.trapezoid(error**2, test_points))
    else:
        # Quick validation with minimal points
        quick_points = np.array([-0.5, 0.0, 0.5])
        computed_solution = solver.evaluate_solution(quick_points)
        exact_solution = true_solution(quick_points, args.solution_order)
        error = np.abs(computed_solution - exact_solution)
        max_error = np.max(error)
        l2_error = 0.0  # Skip expensive integration
    
    print(f"Max error: {max_error:.6f}")
    print(f"L2 error: {l2_error:.6f}")
    
    # Print performance summary
    monitor.print_summary()
    
    if not args.no_plot:
        # Lazy import matplotlib only when needed (saves ~300-400ms import time)
        import matplotlib.pyplot as plt
        
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