from init import Rosenbrock
import numpy as np
from init import grad_f
from init import plot_rosenbrock_with_points

x0 = (-2.00, -2.00)  # Initial point

# Generate a grid for visualization
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = Rosenbrock(X1, X2)

def armijo_gradient_descent(f, grad_f, x0, epsilon=1e-6, max_iter=1000, c1=1e-4, beta=0.5):
    """
    Perform gradient descent with Armijo backtracking line search in 2D space.
    """
    x_k = np.array(x0)
    visited_points = [x_k.copy()]  # Track visited points
    k = 0

    while k < max_iter:
        grad = np.array(grad_f(x_k[0], x_k[1]))
        grad_norm = np.linalg.norm(grad)

        if grad_norm < epsilon:
            print(f"Converged in {k} iterations.")
            break

        alpha_k = 1.0

        # Armijo backtracking line search
        while f(*(x_k - alpha_k * grad)) > f(*x_k) - c1 * alpha_k * np.dot(grad, grad):
            alpha_k *= beta

        x_k = x_k - alpha_k * grad
        visited_points.append(x_k.copy())

        k += 1

    return x_k, visited_points

# Run Armijo gradient descent
opt_point, visited_points = armijo_gradient_descent(Rosenbrock, grad_f, x0, epsilon=1e-6)

# Prepare points for plotting
trial_points = [list(x) for x in visited_points]

# Plot Rosenbrock function and points visited by Armijo gradient descent
plot_rosenbrock_with_points(X1, X2, Z, trial_points, trial_num=1, type="armijo_backtracking")
