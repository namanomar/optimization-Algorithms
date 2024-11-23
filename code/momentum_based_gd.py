from init import Rosenbrock, grad_f, plot_rosenbrock_with_points
import numpy as np


x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = Rosenbrock(X1, X2)

def momentum_gradient_descent(f, grad_f, x0, alpha=0.01, gamma=0.9, epsilon=1e-6, max_iter=1000):
    """
    Perform momentum-based gradient descent in 2D space.

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
    v_k = np.zeros_like(x_k)  # Initial momentum vector
    visited_points = [x_k.copy()]
    k = 0

    while k < max_iter:
        grad = np.array(grad_f(x_k[0], x_k[1]))
        grad_norm = np.linalg.norm(grad)

        # Check stopping criterion
        if grad_norm < epsilon:
            print(f"Converged in {k} iterations.")
            break

        # Update momentum
        v_k = gamma * v_k + alpha * grad
        # Update position
        x_k = x_k - v_k
        visited_points.append(x_k.copy())

        k += 1

    return x_k, visited_points

# Set parameters
x0 = (1.0, -1.7)  # Initial point
alpha = 0.001  # Learning rate
gamma = 0.9    # Momentum coefficient

# Run Momentum-based Gradient Descent
opt_point, visited_points = momentum_gradient_descent(Rosenbrock, grad_f, x0, alpha=alpha, gamma=gamma, epsilon=1e-6)

# Prepare points for plotting
trial_points = [list(x) for x in visited_points]

# Plot Rosenbrock function and points visited by Momentum Gradient Descent
plot_rosenbrock_with_points(X1, X2, Z, trial_points, trial_num=1, type="momentum_gd")
