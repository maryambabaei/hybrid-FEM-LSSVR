import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre, leggauss
from skfem import *
from skfem.helpers import dot, grad

def true_solution(x):
    return np.sin(np.pi * x)

def poisson_rhs(x):
    return np.pi**2 * np.sin(np.pi * x)

def main_boundary_condition_left(x):
    return 0.0  # u(-1) = 0

def main_boundary_condition_right(x):
    return 0.0  # u(1) = 0

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
    training_points = np.linspace(xmin, xmax, 12)
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
        pde_constraints = [residual(u, x) + e[i] for i, x in enumerate(training_points)]
        
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
        
        return np.array(pde_constraints + bc_constraints)
    
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
        
        # Get node values
        interpolator = basis.interpolator(u_fem)
        self.fem_nodes = m.p[0]
        self.fem_values = interpolator(self.fem_nodes.reshape(1, -1)).flatten()
        
        return u_fem, basis
    
    def solve_lssvr_subproblems(self):
        """Solve LSSVR with primal method in each element."""
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
                lssvr_func = lssvr_primal(
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
    
    def solve(self):
        """Complete solution: FEM + LSSVR."""
        self.solve_fem()
        self.solve_lssvr_subproblems()

    
    def evaluate_solution(self, x_points):
        """Evaluate the hybrid solution at given points."""
        solution = np.zeros_like(x_points)
        
        for i, xi in enumerate(x_points):
            # Find which element xi belongs to
            for j in range(len(self.fem_nodes) - 1):
                if self.fem_nodes[j] <= xi <= self.fem_nodes[j + 1]:
                    if callable(self.lssvr_functions[j]):
                        solution[i] = self.lssvr_functions[j](xi)
                    else:
                        # Polynomial object
                        solution[i] = self.lssvr_functions[j](xi)
                    break
            else:
                # Handle boundary cases
                if xi < self.fem_nodes[0]:
                    if callable(self.lssvr_functions[0]):
                        solution[i] = self.lssvr_functions[0](xi)
                    else:
                        solution[i] = self.lssvr_functions[0](xi)
                elif xi > self.fem_nodes[-1]:
                    if callable(self.lssvr_functions[-1]):
                        solution[i] = self.lssvr_functions[-1](xi)
                    else:
                        solution[i] = self.lssvr_functions[-1](xi)
        
        return solution


if __name__ == "__main__":
    # Parameters
    num_nodes = 25
    test_points = np.linspace(-1, 1, 201)
    
    # Solve using hybrid method with primal LSSVR
    solver = FEMLSSVRPrimalSolver(num_nodes, lssvr_M=8, lssvr_gamma=1e4, global_domain=(-1, 1))
    solver.solve()
    
    # Evaluate solution
    computed_solution = solver.evaluate_solution(test_points)
    exact_solution = true_solution(test_points)
    
    
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