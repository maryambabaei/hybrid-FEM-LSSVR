import numpy as np
cimport numpy as cnp
from libc.math cimport pow, sqrt

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

# Initialize numpy
cnp.import_array()

cdef double legendre_p(int n, double x):
    """Evaluate Legendre polynomial P_n(x) using recurrence."""
    cdef double p_prev2, p_prev1, p_current
    cdef int k

    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        p_prev2 = 1.0  # P_0
        p_prev1 = x     # P_1
        for k in range(2, n+1):
            p_current = ((2*k-1) * x * p_prev1 - (k-1) * p_prev2) / k
            p_prev2 = p_prev1
            p_prev1 = p_current
        return p_prev1

cdef double legendre_p_deriv2_robust(int n, double x):
    """
    Compute P_n''(x) using robust recurrence relations with numerical safeguards.
    Uses the three-term recurrence for derivatives directly, with overflow protection.
    """
    cdef double p0 = 1.0      # P_0(x)
    cdef double p1 = x        # P_1(x)
    cdef double d0 = 0.0      # P_0'(x)
    cdef double d1 = 1.0      # P_1'(x)
    cdef double dd0 = 0.0     # P_0''(x)
    cdef double dd1 = 0.0     # P_1''(x)
    
    cdef double pn, dn, ddn
    cdef int k
    cdef double max_val = 1e100  # Overflow threshold
    
    if n == 0:
        return 0.0
    elif n == 1:
        return 0.0
    elif n == 2:
        return 3.0
    else:
        # Use stable recurrence relations for all n > 2
        for k in range(2, n+1):
            # Three-term recurrence for Legendre polynomials:
            # P_k(x) = ((2k-1)*x*P_{k-1}(x) - (k-1)*P_{k-2}(x)) / k
            
            pn = ((2.0*k-1.0)*x*p1 - (k-1.0)*p0) / k
            
            # Check for overflow and rescale if needed
            if abs(pn) > max_val:
                # Rescale all values to prevent overflow
                p0 /= max_val
                p1 /= max_val
                d0 /= max_val
                d1 /= max_val
                dd0 /= max_val
                dd1 /= max_val
                pn /= max_val
            
            # Derivative recurrence:
            # P_k'(x) = ((2k-1)*(P_{k-1}(x) + x*P_{k-1}'(x)) - (k-1)*P_{k-2}'(x)) / k
            
            dn = ((2.0*k-1.0)*(p1 + x*d1) - (k-1.0)*d0) / k
            
            # Second derivative recurrence:
            # P_k''(x) = ((2k-1)*(2*P_{k-1}'(x) + x*P_{k-1}''(x)) - (k-1)*P_{k-2}''(x)) / k
            
            ddn = ((2.0*k-1.0)*(2.0*d1 + x*dd1) - (k-1.0)*dd0) / k
            
            # Update for next iteration
            p0 = p1
            p1 = pn
            d0 = d1
            d1 = dn
            dd0 = dd1
            dd1 = ddn
        
        return dd1

cdef double legendre_p_deriv2(int n, double x):
    """Evaluate second derivative of Legendre polynomial P_n''(x) on [-1,1]."""
    # Use robust recurrence relations for all n to ensure numerical stability
    return legendre_p_deriv2_robust(n, x)

cdef double legendre_p_deriv2_domain(int n, double x, double a, double b):
    """Evaluate second derivative of Legendre polynomial P_n''(x) on domain [a,b]."""
    # Transform x to [-1, 1] domain
    cdef double x_scaled = 2 * (x - a) / (b - a) - 1
    # Scaling factor for derivatives: d/dx = d/dx_scaled * dx_scaled/dx = d/dx_scaled * 2/(b-a)
    # For second derivative: d²/dx² = [2/(b-a)]² * d²/dx_scaled²
    cdef double scale_factor = 2.0 / (b - a)
    cdef double scale_factor2 = scale_factor * scale_factor
    
    # Evaluate second derivative of standard Legendre polynomial
    cdef double d2p_standard = legendre_p_deriv2(n, x_scaled)
    
    return d2p_standard * scale_factor2

cdef double legendre_p_domain(int n, double x, double a, double b):
    """Evaluate Legendre polynomial P_n(x) on domain [a,b]."""
    # Transform x to [-1, 1] domain
    cdef double x_scaled = 2 * (x - a) / (b - a) - 1
    return legendre_p(n, x_scaled)

def build_legendre_matrices_cython(int M, cnp.ndarray[cnp.float64_t, ndim=1] training_points,
                                  double xmin, double xmax, domain_range):
    """
    Cython-optimized version of build_legendre_matrices.
    """
    cdef int n_points = len(training_points)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] A = np.zeros((n_points, M))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] C = np.zeros((2, M))

    cdef double a = domain_range[0]
    cdef double b = domain_range[1]

    # No transformation needed - points are already in the domain
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_points = training_points.copy()
    cdef double xmin_point = xmin
    cdef double xmax_point = xmax

    cdef int i, j

    # Build A matrix (second derivatives at training points)
    for i in range(M):
        for j in range(n_points):
            A[j, i] = -legendre_p_deriv2_domain(i, x_points[j], a, b)

    # Build C matrix (boundary values)
    for i in range(M):
        C[0, i] = legendre_p_domain(i, xmin_point, a, b)
        C[1, i] = legendre_p_domain(i, xmax_point, a, b)

    return A, C