from init import Rosenbrock
import numpy as np
from init import grad_f
from init import plot_rosenbrock_with_points


def nesterov_gradient_descent(f, grad_f, x0, alpha=0.01, gamma=0.9, epsilon=1e-6, max_iter=1000):
    """
    Perform Nesterov Accelerated Gradient Descent (NAG) on a smooth function.
    
    Parameters:
    - f: function, a smooth loss function f: ℝ^n → ℝ
    - grad_f: function, the gradient of f: ℝ^n → ℝ^n
    - x0: array-like, initial guess for the optimization solution (x1, x2)
    - alpha: float, learning rate
    - gamma: float, momentum coefficient in (0,1)
    - epsilon: float, tolerance for stopping criterion
    - max_iter: int, maximum number of iterations

    Returns:
    - x_star: array, local minima of the loss function (x1*, x2*)
    - visited_points: list of points visited during the optimization
    """
    x_k = np.array(x0)
    v_k = np.zeros_like(x_k)  # Initialize momentum vector
    visited_points = [x_k.copy()]
    k = 0

    while k < max_iter:
        # Look-ahead position
        x_lookahead = x_k - gamma * v_k
        # Compute gradient at look-ahead position
        grad = np.array(grad_f(x_lookahead[0], x_lookahead[1]))
        grad_norm = np.linalg.norm(grad)

        # Check stopping criterion
        if grad_norm < epsilon:
            print(f"Converged in {k} iterations.")
            break

        # Update momentum with gradient at look-ahead position
        v_k = gamma * v_k + alpha * grad
        # Update position
        x_k = x_k - v_k
        visited_points.append(x_k.copy())

        k += 1

    return x_k, visited_points

# Set initial conditions and parameters
x0 = (-0.5, -2.0)  # Initial point
alpha = 0.01  # Learning rate
gamma = 0.65    # Momentum coefficient

# Run Nesterov Accelerated Gradient Descent with the modified Rosenbrock function
opt_point, visited_points = nesterov_gradient_descent(Rosenbrock, grad_f, x0, alpha=alpha, gamma=gamma, epsilon=1e-6)

# Prepare for plotting
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = Rosenbrock(X1, X2)

# Convert visited points to list format
trial_points = [list(x) for x in visited_points]

# Plot the result
plot_rosenbrock_with_points(X1, X2, Z, trial_points, trial_num=1, type="nesterov_gd_modified")
