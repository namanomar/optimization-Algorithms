from init import Rosenbrock
import numpy as np
from init import grad_f
from init import plot_rosenbrock_with_points

def adagrad(f, grad_f, x0, epsilon=1e-6, max_iter=1000, alpha=0.1, epsilon_adagrad=1e-8):
    """
    Perform Adagrad optimization in 2D space.
    """
    x_k = np.array(x0)  # Initial point
    G_k = np.zeros_like(x_k)  # Accumulated squared gradients
    visited_points = [x_k.copy()]  # Track visited points
    k = 0

    while k < max_iter:
        grad = np.array(grad_f(x_k[0], x_k[1]))  # Gradient at current point
        grad_norm = np.linalg.norm(grad)

        if grad_norm < epsilon:
            print(f"Converged in {k} iterations.")
            break

        # Update accumulated squared gradients
        G_k += grad**2

        # Update weights using Adagrad
        x_k = x_k - (alpha / (np.sqrt(G_k) + epsilon_adagrad)) * grad
        visited_points.append(x_k.copy())

        k += 1

    return x_k, visited_points

x0 = (-2.00, -2.00)  # Initial point

# Generate a grid for visualization
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = Rosenbrock(X1, X2)


# Run Adagrad optimization
opt_point_adagrad, visited_points_adagrad = adagrad(Rosenbrock, grad_f, x0, epsilon=1e-6)

# Prepare points for plotting
trial_points_adagrad = [list(x) for x in visited_points_adagrad]

# Plot Rosenbrock function and points visited by Adagrad
plot_rosenbrock_with_points(X1, X2, Z, trial_points_adagrad, trial_num=1, type="adagrad")
print(trial_points_adagrad[-1])