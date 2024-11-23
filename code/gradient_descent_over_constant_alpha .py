from matplotlib import pyplot as plt
import numpy as np
from init import Rosenbrock
from init import plot_rosenbrock_with_points
from init import grad_f

def gradient_descent(f, grad_f, x0, epsilon=1e-6, max_iters=1000, alpha=0.01):
    """
    Parameters:
    - f: The function to minimize.
    - grad_f: The gradient of f, a function returning (df/dx1, df/dx2).
    - x0: The initial guess as a tuple (x1, x2).
    - epsilon: Stopping threshold for the gradient norm.
    - max_iters: Maximum number of iterations.
    - alpha: Step size (learning rate).

    Returns:
    - x: Approximation of the local minimum as a tuple (x1, x2).
    """
    x1, x2 = x0  # Start with the initial guess
    k = 0
    visited_points = []

    while k < max_iters:
        # Compute the gradient at the current point
        grad_x1, grad_x2 = grad_f(x1, x2)
        grad_norm = np.sqrt(grad_x1**2 + grad_x2**2)  # Norm of the gradient

        # Stopping criterion: gradient norm is small enough
        if grad_norm < epsilon:
            print(f"Converged in {k} iterations.")
            break

        x1 = x1 - alpha * grad_x1
        x2 = x2 - alpha * grad_x2

        x = (x1, x2)
        k += 1
        visited_points.append(x)


    return x, visited_points

x0 = (1.00, -1.70) #guess or initial point

solution1,points1 = gradient_descent(Rosenbrock, grad_f, x0, epsilon=1e-6, alpha=0.022)  # alpha = 0.022

solution2,points2 = gradient_descent(Rosenbrock, grad_f, x0, epsilon=1e-6, alpha=0.01)  # alpha = 0.01

solution3,points3 = gradient_descent(Rosenbrock, grad_f, x0, epsilon=1e-6, alpha=0.02)  # alpha = 0.02



x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = Rosenbrock(X1, X2)

trials = [list(points1), list(points2), list(points3)]

for i, trial in enumerate(trials, start=1):
    plot_rosenbrock_with_points(X1, X2, Z, trial, trial_num=i,type="Gradient Descent with constant alpha")