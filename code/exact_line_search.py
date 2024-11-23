from init import Rosenbrock
import numpy as np
from init import grad_f
from init import plot_rosenbrock_with_points
x0 = (1.00, -1.70) #guess or initial point

x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = Rosenbrock(X1, X2)

def gradient_descent_exact_line_search(f, grad_f, x0, epsilon=1e-6, max_iter=1000):
    """
    Perform gradient descent with exact line search in 2D space.

    Parameters:
    - f: function, a smooth loss function f: ℝ2 → ℝ
    - grad_f: function, the gradient of f: ℝ2 → ℝ2, returns (df/dx1, df/dx2)
    - x0: array-like, initial guess for the optimization solution (x1, x2)
    - epsilon: float, tolerance for stopping criterion
    - max_iter: int, maximum number of iterations to prevent infinite loops

    Returns:
    - x_star: array, local minima of the loss function (x1*, x2*)
    - visited_points: list of points visited during the optimization
    """
    x_k = np.array(x0)
    visited_points = [x_k.copy()]  # Initialize visited points list with the starting point
    k = 0

    while k < max_iter:
        # Compute the gradient at the current point
        grad = np.array(grad_f(x_k[0], x_k[1]))
        norm_grad = np.linalg.norm(grad)

        # Check stopping criteria
        if norm_grad < epsilon:
            break

        # Exact line search: find alpha that minimizes f(x_k - alpha * grad)
        alpha_k = minimize_line_search(f, x_k, grad)

        # Update the components
        x_k = x_k - alpha_k * grad

        # Append the new position to the visited points
        visited_points.append(x_k.copy())

        k += 1

    return x_k, visited_points  # Return both the local minimum and the visited points

def minimize_line_search(f, x_k, grad):
    """
    Perform exact line search to find the optimal alpha.

    Parameters:
    - f: function, the objective function
    - x_k: array-like, current position in the optimization
    - grad: array-like, gradient at current position

    Returns:
    - alpha: optimal step size that minimizes f along the gradient direction
    """
    # Define the function to minimize
    def line_search_func(alpha):
        return f(x_k[0] - alpha * grad[0], x_k[1] - alpha * grad[1])

    # Use a simple grid search or optimization method to find the minimum
    alpha_values = np.linspace(0, 1, 100)  # Try values between 0 and 1
    f_values = [line_search_func(alpha) for alpha in alpha_values]

    # Find the alpha that gives the minimum function value
    min_index = np.argmin(f_values)
    return alpha_values[min_index]

outer, points = gradient_descent_exact_line_search(Rosenbrock, grad_f, x0, epsilon=1e-6)

trial = [list(x) for x in points]

plot_rosenbrock_with_points(X1, X2, Z, trial, trial_num=1,type="exact_line_search")