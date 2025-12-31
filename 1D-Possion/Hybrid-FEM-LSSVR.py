import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre, leggauss
from skfem import *
from skfem.helpers import dot, grad
import argparse
import time
import psutil
import os

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

def true_solution(x):
    return np.sin(np.pi * x)

def poisson_rhs(x):
    return np.pi**2 * np.sin(np.pi * x)

def main_boundary_condition_left(x):
    return 0.0  # u(-1) = 0

def main_boundary_condition_right(x):
    return 0.0  # u(1) = 0

def build_legendre_matrices(M, training_points, xmin, xmax, domain_range):
    """
    Build matrices for Legendre polynomial constraints.
    
    Returns:
    A: Matrix for PDE constraints (second derivatives at training points)
    C: Matrix for boundary constraints (function values at boundaries)
    """
    # PDE constraint matrix: A * w = -u''(training_points)
    A = np.zeros((len(training_points), M))
    for i in range(M):
        w = np.zeros(M)
        w[i] = 1.0
        u = Legendre(w, domain_range)
        A[:, i] = -u.deriv(2)(training_points)
    
    # Boundary constraint matrix: C * w = [u(xmin), u(xmax)]
    C = np.zeros((2, M))
    for i in range(M):
        w = np.zeros(M)
        w[i] = 1.0
        u = Legendre(w, domain_range)
        C[0, i] = u(xmin)
        C[1, i] = u(xmax)
    
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
    A, C = build_legendre_matrices(M, training_points, xmin, xmax, domain_range)
    
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
    
    kkt_matrix = np.block([
        [np.eye(M) + gamma * A.T @ A, C.T],
        [C, np.zeros((2, 2))]
    ])
    
    kkt_rhs = np.concatenate([
        gamma * A.T @ b_pde,
        b_bc
    ])
    
    # Solve the system
    try:
        solution = solve(kkt_matrix, kkt_rhs)
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
        print("Warning: Linear system singular, using optimization fallback")
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
    initial = np.concatenate([np.random.rand(M) * 0.01, np.zeros(n_interior)])
    
    # Equality constraints
    cons = {'type': 'eq', 'fun': constraints}
    
    # Solve optimization problem
    res = minimize(objective, x0=initial, constraints=cons, method='SLSQP', 
                   options={'maxiter': 1000, 'ftol': 1e-12})
    
    if not res.success:
        print(f"Warning: Optimization may not have converged: {res.message}")
        print(f"Final constraint violation: {np.max(np.abs(constraints(res.x)))}")
    
    # Extract solution
    u_lssvr = Legendre(res.x[:M], domain_range)
    slack_variables = res.x[M:M+n_interior]
    
    # Print some diagnostics
    max_slack = np.max(np.abs(slack_variables))
    constraint_violation = np.max(np.abs(constraints(res.x)))
    
    return u_lssvr

class FEMLSSVRPrimalSolver:
    def __init__(self, num_fem_nodes=5, lssvr_M=12, lssvr_gamma=1e6, global_domain=(-1, 1)):
        self.num_fem_nodes = num_fem_nodes
        self.lssvr_M = lssvr_M
        self.lssvr_gamma = lssvr_gamma
        self.global_domain = global_domain
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
            return -dot(grad(u), grad(v))
        
        @LinearForm
        def load(v, w):
            x = w.x[0]
            return -(np.pi**2) * np.sin(np.pi * x) * v
        
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
        
        return u_fem, basis
    
    def solve_lssvr_subproblems(self):
        """Solve LSSVR with primal method in each element."""
        lssvr_start = time.time()
        monitor.memory_usage.append(monitor.get_memory_usage())
        
        self.lssvr_functions = []
        
        for i in range(len(self.fem_nodes) - 1):
            x_start = self.fem_nodes[i]
            x_end = self.fem_nodes[i + 1]
            u_left = self.fem_values[i]
            u_right = self.fem_values[i + 1]
            
            # Check if this element is at domain boundaries
            is_left_boundary = (i == 0)  # First element
            is_right_boundary = (i == len(self.fem_nodes) - 2)  # Last element
            
            # Solve LSSVR with primal method for this element
            try:
                lssvr_func = lssvr_primal_direct(
                    poisson_rhs, [x_start, x_end], u_left, u_right, 
                    self.lssvr_M, self.lssvr_gamma,
                    is_left_boundary=is_left_boundary,
                    is_right_boundary=is_right_boundary,
                    global_domain_range=self.global_domain
                )
                self.lssvr_functions.append(lssvr_func)
            except Exception as e:
                print(f"Error in element {i+1}: {e}")
                # Fallback: use linear interpolation
                def linear_fallback(x):
                    return u_left + (u_right - u_left) * (x - x_start) / (x_end - x_start)
                self.lssvr_functions.append(linear_fallback)
        
        lssvr_time = time.time() - lssvr_start
        monitor.lssvr_total_time = lssvr_time
        monitor.record_operation('LSSVR_subproblems', lssvr_time, f'elements={len(self.lssvr_functions)}')
        monitor.memory_usage.append(monitor.get_memory_usage())
    
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
    args = parser.parse_args()
    
    # Initialize performance monitoring
    monitor.reset()
    
    # Parameters
    num_nodes = 25
    test_points = np.linspace(-1, 1, 201)
    
    # Solve using hybrid method with primal LSSVR
    solver = FEMLSSVRPrimalSolver(num_nodes, lssvr_M=8, lssvr_gamma=1e4, global_domain=(-1, 1))
    solver.solve()
    
    # Evaluate solution
    computed_solution = solver.evaluate_solution(test_points)
    exact_solution = true_solution(test_points)
    
    # Calculate errors
    error = np.abs(computed_solution - exact_solution)
    max_error = np.max(error)
    l2_error = np.sqrt(np.trapz(error**2, test_points))
    print(f"Max error: {max_error:.6f}")
    print(f"L2 error: {l2_error:.6f}")
    
    # Print performance summary
    monitor.print_summary()
    
    if not args.no_plot:
        # plot of solution
        plt.figure(figsize=(10, 6))
        plt.plot(test_points, exact_solution, 'r-', label='Exact Solution', linewidth=2)
        plt.plot(test_points, computed_solution, 'b--', label='FEM+LSSVR Solution', linewidth=2)
        plt.scatter(solver.fem_nodes, solver.fem_values, c='green', s=50, label='FEM Nodes', zorder=5)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
        plt.show()