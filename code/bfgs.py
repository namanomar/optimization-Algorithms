import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm, cholesky, LinAlgError
from scipy.optimize import line_search

from newton_method import damped_newton, gradient_descent_newton

def bfgs_optimize(f, grad_f, x0, epsilon=1e-6, max_iter=100):
    """
    BFGS optimization algorithm

    Args:
        f (callable): Objective function
        grad_f (callable): Gradient function
        x0 (np.ndarray): Initial point
        epsilon (float): Convergence criterion
        max_iter (int): Maximum iterations

    Returns:
        np.ndarray: Path of optimization points
    """
    n = len(x0)
    x_k = x0.copy()
    H_k = np.eye(n)
    g_k = grad_f(x_k)
    path = [x_k.copy()]

    for k in range(max_iter):
        if norm(g_k) < epsilon:
            break

        # Compute search direction
        p_k = -H_k @ g_k

        # Line search
        alpha = line_search(f, grad_f, x_k, p_k)[0]
        if alpha is None:
            alpha = 0.1

        # Update position
        x_k1 = x_k + alpha * p_k
        g_k1 = grad_f(x_k1)

        # Compute differences
        s_k = x_k1 - x_k
        y_k = g_k1 - g_k

        # BFGS update
        try:
            rho_k = 1.0 / (y_k @ s_k)
            I = np.eye(n)
            H_k = (I - rho_k * np.outer(s_k, y_k)) @ H_k @ (I - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k)
        except:
            # If update fails, reset Hessian approximation
            H_k = np.eye(n)

        x_k = x_k1
        g_k = g_k1
        path.append(x_k.copy())

    return np.array(path)

def plot_optimization_paths(f, grad_f, hess_f, x0, bounds, title, functions_dict,id):
    """
    Plot optimization comparison between different methods

    Args:
        f (callable): Objective function
        grad_f (callable): Gradient function
        hess_f (callable): Hessian function
        x0 (np.ndarray): Initial point
        bounds (tuple): Plot bounds ((x_min, x_max), (y_min, y_max))
        title (str): Plot title
        functions_dict (dict): Dictionary of optimization functions to compare
    """
    plt.figure(figsize=(12, 8))

    # Create contour plot
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = f(np.array([X[i,j], Y[i,j]]))

    # Plot contours
    plt.contour(X, Y, Z, levels=50, colors='lightgray', alpha=0.6)

    # Colors for different methods
    colors = ['b', 'r', 'g', 'y']

    # Run and plot each optimization method
    for (method_name, method_func), color in zip(functions_dict.items(), colors):
        try:
            if method_name == "Damped Newton":
                path = method_func(f, grad_f, hess_f, x0)
            elif method_name == "Gradient Descent":
                path = method_func(grad_f, x0)[0]  # Note: Original function returns tuple
            else:
                path = method_func(f, grad_f, x0)

            plt.plot(path[:,0], path[:,1], f'{color}.-',
                    label=f'{method_name} ({len(path)} iter)',
                    linewidth=1.5, markersize=4)
        except Exception as e:
            print(f"Error running {method_name}: {e}")

    # Plot initial point
    plt.plot(x0[0], x0[1], 'ko', label='Initial point')

    # Formatting
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def f1(x):
    return 0.5 * (2*x[0]**2 + 3*x[1]**2) - (x[0] + x[1])

def grad_f1(x):
    return np.array([2*x[0] - 1, 3*x[1] - 1])

def hess_f1(x):
    return np.array([[2, 0], [0, 3]])

def f2(x):
    return 0.5 * (2*x[0]**2 + 20*x[1]**2) - (x[0] + x[1])

def grad_f2(x):
    return np.array([2*x[0] - 1, 20*x[1] - 1])

def hess_f2(x):
    return np.array([[2, 0], [0, 20]])

def f3(x):
    return (1 - x[0])**2 + 10*(x[1] - x[0]**2)**2

def grad_f3(x):
    return np.array([
        -2*(1 - x[0]) - 40*x[0]*(x[1] - x[0]**2),
        20*(x[1] - x[0]**2)
    ])

def hess_f3(x):
    return np.array([
        [120*x[0]**2 - 40*x[1] + 2, -40*x[0]],
        [-40*x[0], 20]
    ])

x0 = np.array([-1.2, 1.0])
bounds = ((-3, 3), (-1, 3))

optimization_methods = {
    "BFGS": lambda f, g, x0: bfgs_optimize(f, g, x0),
    "Damped Newton": lambda f, g, h, x0: damped_newton(f, g, h, x0),
    "Gradient Descent": lambda g, x0: gradient_descent_newton(g, x0)
}

# Plot all three functions
test_functions = [
    (f1, grad_f1, hess_f1, "Function f₁(x,y)"),
    (f2, grad_f2, hess_f2, "Function f₂(x,y)"),
    (f3, grad_f3, hess_f3, "Function f₃(x,y)")
]
i=0
for f, grad_f, hess_f, title in test_functions:
    plot_optimization_paths(f, grad_f, hess_f, x0, bounds, title, optimization_methods,i)
    i+=1