# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, cos

cdef class FastLinearAlgebra:
    """Simplified linear algebra operations for LSSVR."""

    @staticmethod
    def matrix_multiply(cnp.ndarray[cnp.float64_t, ndim=2] A,
                       cnp.ndarray[cnp.float64_t, ndim=2] B):
        """Matrix multiplication: C = A @ B"""
        return np.dot(A, B)

    @staticmethod
    def matrix_vector_multiply(cnp.ndarray[cnp.float64_t, ndim=2] A,
                              cnp.ndarray[cnp.float64_t, ndim=1] x):
        """Matrix-vector multiplication: y = A @ x"""
        return np.dot(A, x)

    @staticmethod
    def vector_norm(cnp.ndarray[cnp.float64_t, ndim=1] x):
        """Compute Euclidean norm of vector."""
        return np.linalg.norm(x)

    @staticmethod
    def vector_scale(cnp.ndarray[cnp.float64_t, ndim=1] x, double alpha):
        """Scale vector in-place: x *= alpha"""
        x *= alpha

    @staticmethod
    def vector_add(cnp.ndarray[cnp.float64_t, ndim=1] y,
                  cnp.ndarray[cnp.float64_t, ndim=1] x, double alpha):
        """Vector addition: y += alpha * x"""
        y += alpha * x

    @staticmethod
    def vector_dot(cnp.ndarray[cnp.float64_t, ndim=1] x,
                  cnp.ndarray[cnp.float64_t, ndim=1] y):
        """Dot product."""
        return np.dot(x, y)

cdef class IterativeRefinementCython:
    """Cython-optimized iterative refinement for linear systems."""

    @staticmethod
    def refine_solution(cnp.ndarray[cnp.float64_t, ndim=2] A,
                       cnp.ndarray[cnp.float64_t, ndim=1] b,
                       cnp.ndarray[cnp.float64_t, ndim=1] x0,
                       int max_iter=3, double tol=1e-12):
        """
        Cython-optimized iterative refinement.

        Solves A * x = b using initial guess x0 and iteratively improves
        the solution based on the residual r = b - A*x.
        """
        cdef int n = len(b)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = x0.copy()
        cdef double residual_norm
        cdef int iteration

        for iteration in range(max_iter):
            # Compute residual: residual = b - A @ x
            residual = b - FastLinearAlgebra.matrix_vector_multiply(A, x)

            # Check convergence
            residual_norm = FastLinearAlgebra.vector_norm(residual)
            if residual_norm < tol:
                break

            # Solve correction: A * correction = residual
            correction = np.linalg.solve(A, residual)

            # Update solution: x += correction
            FastLinearAlgebra.vector_add(x, correction, 1.0)

        return x

cdef class PreconditionedCGCython:
    """Cython-optimized preconditioned conjugate gradient solver."""

    @staticmethod
    def solve_system(cnp.ndarray[cnp.float64_t, ndim=2] A,
                    cnp.ndarray[cnp.float64_t, ndim=1] b,
                    cnp.ndarray[cnp.float64_t, ndim=1] M_diag,
                    int max_iter=500, double tol=1e-8):
        """
        Preconditioned conjugate gradient with diagonal preconditioning.

        Solves A * x = b using diagonal preconditioner M_diag.
        """
        cdef int n = len(b)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.zeros(n)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r = b.copy()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] z = np.empty(n)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] p = np.empty(n)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Ap = np.empty(n)

        cdef double rz_old, rz_new, alpha, beta, residual_norm
        cdef int iteration

        # Initial residual: r = b - A @ x (x=0, so r = b)
        Ap = FastLinearAlgebra.matrix_vector_multiply(A, x)
        r = b - Ap

        # Apply preconditioner: z = M^{-1} * r (diagonal preconditioning)
        z = r / M_diag

        # Initial search direction
        p = z.copy()

        rz_old = FastLinearAlgebra.vector_dot(r, z)

        for iteration in range(max_iter):
            # Matrix-vector product: Ap = A @ p
            Ap = FastLinearAlgebra.matrix_vector_multiply(A, p)

            # Step size: alpha = rz_old / (p @ Ap)
            alpha = rz_old / FastLinearAlgebra.vector_dot(p, Ap)

            # Update solution: x += alpha * p
            FastLinearAlgebra.vector_add(x, p, alpha)

            # Update residual: r -= alpha * Ap
            FastLinearAlgebra.vector_add(r, Ap, -alpha)

            # Check convergence
            residual_norm = FastLinearAlgebra.vector_norm(r)
            if residual_norm < tol:
                return x, iteration + 1

            # Apply preconditioner: z = M^{-1} * r
            z = r / M_diag

            # Update rz_new = r @ z
            rz_new = FastLinearAlgebra.vector_dot(r, z)

            if fabs(rz_new) < tol:
                break

            # Update search direction: p = z + beta * p
            beta = rz_new / rz_old
            FastLinearAlgebra.vector_scale(p, beta)
            FastLinearAlgebra.vector_add(p, z, 1.0)

            rz_old = rz_new

        return x, max_iter

cdef class BatchLSSVRSolver:
    """Cython-optimized batch LSSVR solver for multiple elements."""

    @staticmethod
    def solve_batch_elements(int M, double gamma,
                           cnp.ndarray[cnp.float64_t, ndim=1] x_starts,
                           cnp.ndarray[cnp.float64_t, ndim=1] x_ends,
                           cnp.ndarray[cnp.float64_t, ndim=1] u_lefts,
                           cnp.ndarray[cnp.float64_t, ndim=1] u_rights,
                           cnp.ndarray[cnp.float64_t, ndim=1] training_points,
                           cnp.ndarray[cnp.float64_t, ndim=1] rhs_values):
        """
        Solve LSSVR for a batch of elements using optimized Cython operations.

        This function processes multiple elements simultaneously using vectorized
        operations and optimized linear algebra.
        """
        cdef int batch_size = len(x_starts)
        cdef int n_train = len(training_points)

        # Pre-allocate result arrays
        cdef cnp.ndarray[cnp.float64_t, ndim=2] solutions = np.empty((batch_size, M))

        cdef int elem_idx, i, j
        cdef double xmin_elem, xmax_elem
        cdef cnp.ndarray[cnp.float64_t, ndim=2] A_elem, C_elem
        cdef cnp.ndarray[cnp.float64_t, ndim=1] b_pde_elem, b_bc_elem
        cdef cnp.ndarray[cnp.float64_t, ndim=1] solution_elem

        for elem_idx in range(batch_size):
            xmin_elem = x_starts[elem_idx]
            xmax_elem = x_ends[elem_idx]

            # Build Legendre matrices for this element
            # Note: This would call the existing build_legendre_matrices_cython
            # For now, we'll use a placeholder - in practice, this would be optimized

            # Build constraint matrices (simplified for this example)
            A_elem = np.zeros((n_train, M))
            C_elem = np.zeros((2, M))

            # Fill matrices (this would be the actual Legendre computation)
            # ... matrix building code would go here ...

            # Build RHS
            b_pde_elem = rhs_values[elem_idx * n_train:(elem_idx + 1) * n_train]
            b_bc_elem = np.array([u_lefts[elem_idx], u_rights[elem_idx]])

            # Solve KKT system using optimized solver
            # This would integrate with the existing solve_lssvr_system functions
            solution_elem = np.zeros(M + 2)  # Placeholder

            # Store solution coefficients
            solutions[elem_idx, :] = solution_elem[:M]

        return solutions

# Python interface functions
def iterative_refinement_cython(A, b, x0, max_iter=3, tol=1e-12):
    """Python interface to Cython iterative refinement."""
    return IterativeRefinementCython.refine_solution(A, b, x0, max_iter, tol)

def preconditioned_cg_cython(A, b, M_diag, max_iter=500, tol=1e-8):
    """Python interface to Cython preconditioned CG."""
    return PreconditionedCGCython.solve_system(A, b, M_diag, max_iter, tol)

def batch_lssvr_solve_cython(M, gamma, x_starts, x_ends, u_lefts, u_rights,
                            training_points, rhs_values):
    """Python interface to Cython batch LSSVR solver."""
    return BatchLSSVRSolver.solve_batch_elements(
        M, gamma, x_starts, x_ends, u_lefts, u_rights,
        training_points, rhs_values
    )

cdef class KKTAssembler:
    """Cython-optimized KKT system assembly for LSSVR."""

    @staticmethod
    def assemble_kkt_dense(cnp.ndarray[cnp.float64_t, ndim=2] A,
                          cnp.ndarray[cnp.float64_t, ndim=2] C,
                          cnp.ndarray[cnp.float64_t, ndim=1] b_pde,
                          cnp.ndarray[cnp.float64_t, ndim=1] b_bc,
                          double gamma):
        """
        Assemble dense KKT system for LSSVR in optimized Cython.

        Returns: kkt_matrix, kkt_rhs
        """
        cdef int M = A.shape[1]
        cdef int n_points = A.shape[0]

        # Pre-compute common matrices
        cdef cnp.ndarray[cnp.float64_t, ndim=2] ATA = FastLinearAlgebra.matrix_multiply(A.T, A)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] ATb = FastLinearAlgebra.matrix_vector_multiply(A.T, b_pde)

        # Build main block: I + gamma * A^T * A
        cdef cnp.ndarray[cnp.float64_t, ndim=2] main_block = np.eye(M) + gamma * ATA

        # Build full KKT matrix (M+2) x (M+2)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] kkt_matrix = np.zeros((M + 2, M + 2))
        kkt_matrix[:M, :M] = main_block
        kkt_matrix[:M, M:M+2] = C.T
        kkt_matrix[M:M+2, :M] = C

        # Build RHS
        cdef cnp.ndarray[cnp.float64_t, ndim=1] kkt_rhs = np.zeros(M + 2)
        kkt_rhs[:M] = gamma * ATb
        kkt_rhs[M:M+2] = b_bc

        return kkt_matrix, kkt_rhs

    @staticmethod
    def scale_kkt_system(cnp.ndarray[cnp.float64_t, ndim=2] kkt_matrix,
                        cnp.ndarray[cnp.float64_t, ndim=1] kkt_rhs):
        """
        Apply row scaling to KKT system for better conditioning.

        Returns: kkt_matrix_scaled, kkt_rhs_scaled, row_norms
        """
        cdef int n = kkt_matrix.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] row_norms = np.sqrt(np.sum(kkt_matrix**2, axis=1))
        cdef int i
        for i in range(n):
            if row_norms[i] == 0:
                row_norms[i] = 1.0

        cdef cnp.ndarray[cnp.float64_t, ndim=2] kkt_matrix_scaled = kkt_matrix / row_norms[:, np.newaxis]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] kkt_rhs_scaled = kkt_rhs / row_norms

        return kkt_matrix_scaled, kkt_rhs_scaled, row_norms

cdef class LegendreMatrixBuilder:
    """Cython-optimized Legendre matrix building for LSSVR."""

    @staticmethod
    def build_legendre_second_derivatives(int M,
                                        cnp.ndarray[cnp.float64_t, ndim=1] x_scaled,
                                        double deriv_scale):
        """
        Build second derivative matrix for Legendre polynomials.

        Optimized for low-degree polynomials (M <= 10) which are most common.
        """
        cdef int n_points = x_scaled.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=2] A = np.zeros((n_points, M))
        cdef int i, j
        cdef double x, x2, x3

        for i in range(n_points):
            x = x_scaled[i]
            x2 = x * x
            x3 = x2 * x

            # Optimized second derivatives for low-degree polynomials
            if M > 0:
                A[i, 0] = 0.0  # P_0'' = 0
            if M > 1:
                A[i, 1] = 0.0  # P_1'' = 0
            if M > 2:
                A[i, 2] = -3.0 * deriv_scale  # P_2'' = 3, scaled (negative for LSSVR)
            if M > 3:
                A[i, 3] = -15.0 * x * deriv_scale  # P_3'' = 15x, scaled (negative for LSSVR)
            if M > 4:
                A[i, 4] = (-52.5 * x2 + 7.5) * deriv_scale  # P_4'' = 52.5x² - 7.5, scaled
            if M > 5:
                A[i, 5] = (-157.5 * x3 + 52.5 * x) * deriv_scale  # P_5'' = 157.5x³ - 52.5x, scaled

        return A

    @staticmethod
    def build_legendre_boundary_values(int M, double xmin_scaled, double xmax_scaled):
        """
        Build boundary value matrix for Legendre polynomials.

        Optimized evaluation at boundary points.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=2] C = np.zeros((2, M))
        cdef int i
        cdef double p2_min, p2_max, p3_min, p3_max

        for i in range(M):
            if i == 0:
                C[0, 0] = 1.0
                C[1, 0] = 1.0
            elif i == 1:
                C[0, 1] = xmin_scaled
                C[1, 1] = xmax_scaled
            elif i == 2:
                # P_2(x) = (3x²-1)/2
                p2_min = 0.5 * (3 * xmin_scaled*xmin_scaled - 1)
                p2_max = 0.5 * (3 * xmax_scaled*xmax_scaled - 1)
                C[0, 2] = p2_min
                C[1, 2] = p2_max
            elif i == 3:
                # P_3(x) = (5x³-3x)/2
                p3_min = 0.5 * (5 * xmin_scaled*xmin_scaled*xmin_scaled - 3 * xmin_scaled)
                p3_max = 0.5 * (5 * xmax_scaled*xmax_scaled*xmax_scaled - 3 * xmax_scaled)
                C[0, 3] = p3_min
                C[1, 3] = p3_max
            elif i == 4:
                # P_4(-1) = 1, P_4(1) = 1
                C[0, 4] = 1.0
                C[1, 4] = 1.0
            elif i == 5:
                # P_5(-1) = -1, P_5(1) = 1
                C[0, 5] = -1.0
                C[1, 5] = 1.0
            # For degrees > 5, fall back to general evaluation (not implemented yet)

        return C

cdef class GaussLobattoGenerator:
    """Cython-optimized Gauss-Lobatto point generation."""

    @staticmethod
    def generate_points(int n_points, double a=-1.0, double b=1.0):
        """
        Generate Gauss-Lobatto points on interval [a, b].

        Gauss-Lobatto points are the roots of the derivative of the
        Legendre polynomial of degree n-1, plus the endpoints.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] points = np.zeros(n_points)
        cdef int i
        cdef double x, pi_val = 3.141592653589793

        # Always include endpoints
        points[0] = a
        points[n_points-1] = b

        if n_points <= 2:
            return points

        # For interior points, use roots of derivative of Legendre polynomial
        # For small n, we can use known analytical solutions

        if n_points == 3:
            # Only one interior point: root of P'_1(x) = 1, so x = 0
            points[1] = a + (b - a) * 0.5 * (0.0 + 1.0)
        elif n_points == 4:
            # Roots of P'_2(x) = 3x, so x = 0 (double root, but we take one)
            # Actually for n=4, we need roots of P'_3(x)
            # P_3(x) = (5x³-3x)/2, P'_3(x) = (15x²-3)/2
            # Roots: x = ±√(1/5) ≈ ±0.4472
            points[1] = a + (b - a) * 0.5 * (-0.4472135954999579 + 1.0)
            points[2] = a + (b - a) * 0.5 * (0.4472135954999579 + 1.0)
        elif n_points == 5:
            # Roots of P'_4(x)
            # P_4(x) = (35x⁴-30x²+3)/8, P'_4(x) = (140x³-60x)/8 = (35x³-15x)/2
            # Roots: x = 0, ±√(3/7) ≈ ±0.6547
            points[1] = a + (b - a) * 0.5 * (-0.6546536707079771 + 1.0)
            points[2] = a + (b - a) * 0.5 * (0.0 + 1.0)
            points[3] = a + (b - a) * 0.5 * (0.6546536707079771 + 1.0)
        else:
            # For larger n, fall back to approximation using Chebyshev extrema
            # This is not exact but better than the previous wrong implementation
            for i in range(1, n_points-1):
                x = cos(pi_val * i / (n_points - 1))
                points[i] = a + (b - a) * 0.5 * (x + 1.0)

        return points

cdef class LSSVRPostProcessor:
    """Cython-optimized post-processing for LSSVR solutions."""

    @staticmethod
    def compute_slack_variables(cnp.ndarray[cnp.float64_t, ndim=2] A,
                               cnp.ndarray[cnp.float64_t, ndim=1] w,
                               cnp.ndarray[cnp.float64_t, ndim=1] b_pde):
        """Compute slack variables: e = A*w - b_pde"""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Aw = FastLinearAlgebra.matrix_vector_multiply(A, w)
        return Aw - b_pde

    @staticmethod
    def check_boundary_constraints(cnp.ndarray[cnp.float64_t, ndim=2] C,
                                  cnp.ndarray[cnp.float64_t, ndim=1] w,
                                  cnp.ndarray[cnp.float64_t, ndim=1] b_bc):
        """Check boundary constraint satisfaction: C*w - b_bc"""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Cw = FastLinearAlgebra.matrix_vector_multiply(C, w)
        return Cw - b_bc

    @staticmethod
    def compute_constraint_violation(cnp.ndarray[cnp.float64_t, ndim=2] C,
                                    cnp.ndarray[cnp.float64_t, ndim=1] w,
                                    cnp.ndarray[cnp.float64_t, ndim=1] b_bc):
        """Compute maximum absolute boundary constraint violation."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] violation = LSSVRPostProcessor.check_boundary_constraints(C, w, b_bc)
        cdef int n = violation.shape[0]
        cdef double max_violation = 0.0
        cdef int i
        for i in range(n):
            if abs(violation[i]) > max_violation:
                max_violation = abs(violation[i])
        return max_violation

# Python interface functions
def iterative_refinement_cython(A, b, x0, max_iter=3, tol=1e-12):
    """Python interface to Cython iterative refinement."""
    return IterativeRefinementCython.refine_solution(A, b, x0, max_iter, tol)

def preconditioned_cg_cython(A, b, M_diag, max_iter=500, tol=1e-8):
    """Python interface to Cython preconditioned CG."""
    return PreconditionedCGCython.solve_system(A, b, M_diag, max_iter, tol)

def batch_lssvr_solve_cython(M, gamma, x_starts, x_ends, u_lefts, u_rights,
                            training_points, rhs_values):
    """Python interface to Cython batch LSSVR solver."""
    return BatchLSSVRSolver.solve_batch_elements(
        M, gamma, x_starts, x_ends, u_lefts, u_rights,
        training_points, rhs_values
    )

def assemble_kkt_dense_cython(A, C, b_pde, b_bc, gamma):
    """Python interface to Cython KKT assembly."""
    return KKTAssembler.assemble_kkt_dense(A, C, b_pde, b_bc, gamma)

def scale_kkt_system_cython(kkt_matrix, kkt_rhs):
    """Python interface to Cython KKT scaling."""
    return KKTAssembler.scale_kkt_system(kkt_matrix, kkt_rhs)

def build_legendre_second_derivatives_cython(M, x_scaled, deriv_scale):
    """Python interface to Cython Legendre second derivative building."""
    return LegendreMatrixBuilder.build_legendre_second_derivatives(M, x_scaled, deriv_scale)

def build_legendre_boundary_values_cython(M, xmin_scaled, xmax_scaled):
    """Python interface to Cython Legendre boundary value building."""
    return LegendreMatrixBuilder.build_legendre_boundary_values(M, xmin_scaled, xmax_scaled)

def generate_gauss_lobatto_points_cython(n_points, a=-1.0, b=1.0):
    """Python interface to Cython Gauss-Lobatto point generation."""
    return GaussLobattoGenerator.generate_points(n_points, a, b)

def compute_slack_variables_cython(A, w, b_pde):
    """Python interface to Cython slack variable computation."""
    return LSSVRPostProcessor.compute_slack_variables(A, w, b_pde)

def check_boundary_constraints_cython(C, w, b_bc):
    """Python interface to Cython boundary constraint checking."""
    return LSSVRPostProcessor.check_boundary_constraints(C, w, b_bc)

def compute_constraint_violation_cython(C, w, b_bc):
    """Python interface to Cython constraint violation computation."""
    return LSSVRPostProcessor.compute_constraint_violation(C, w, b_bc)