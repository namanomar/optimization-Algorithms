# Damped Newton Method
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm, cholesky, LinAlgError
from scipy.optimize import line_search

def damped_newton(f, grad_f, hess_f, x0, lambda_factor=0.1, epsilon=1e-6, max_iters=100):
    x = x0
    path = [x.copy()]
    for _ in range(max_iters):
        grad = grad_f(x)
        if norm(grad) < epsilon:
            break
        H = hess_f(x)
        try:
            # If Hessian is positive definite, perform Cholesky decomposition
            L = cholesky(H + lambda_factor * np.eye(len(x)))  # Damping term to ensure PD
            delta = -np.linalg.solve(L.T, np.linalg.solve(L, grad))
        except LinAlgError:
            # If not PD, add lambda_factor to make it PD and retry
            delta = -np.linalg.solve(H + lambda_factor * np.eye(len(x)), grad)
        x = x + delta
        path.append(x.copy())
    return np.array(path)

# Gradient Descent Method
def gradient_descent_newton(grad_f, x0, alpha=0.1, epsilon=1e-6, max_iters=100):
    x = x0
    path = [x.copy()]
    for _ in range(max_iters):
        grad = grad_f(x)
        if norm(grad) < epsilon:
            break
        x = x - alpha * grad
        path.append(x.copy())
    return np.array(path)

# Plotting function
def plot_contours_and_paths(f, grad_path, newton_path, title, xlims=(-3, 3), ylims=(-3, 3)):
    x = np.linspace(xlims[0], xlims[1], 100)
    y = np.linspace(ylims[0], ylims[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda x, y: f([x, y]))(X, Y)

    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.plot(grad_path[:, 0], grad_path[:, 1], 'r-o', markersize=3, label="GD Path")
    plt.plot(newton_path[:, 0], newton_path[:, 1], 'g-o', markersize=3, label="Newton Path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.colorbar(label="Function Value")

# Define the functions, gradients, and Hessians
def f1(x):  # f1(x, y) = x^2 + y^2
    return x[0]**2 + x[1]**2

def grad_f1(x):
    return np.array([2 * x[0], 2 * x[1]])

def hess_f1(x):
    return np.array([[2, 0], [0, 2]])

def f2(x):  # f2(x, y) = 2*x^2 + 0.5*y^2
    return 2 * x[0]**2 + 0.5 * x[1]**2

def grad_f2(x):
    return np.array([4 * x[0], x[1]])

def hess_f2(x):
    return np.array([[4, 0], [0, 1]])

def f3(x):  # f3(x, y) = x^2 + 10*y^2
    return x[0]**2 + 10 * x[1]**2

def grad_f3(x):
    return np.array([2 * x[0], 20 * x[1]])

def hess_f3(x):
    return np.array([[2, 0], [0, 20]])

def f4(x):  # f4(x, y) = x^2 + 5*y^2
    return x[0]**2 + 5 * x[1]**2

def grad_f4(x):
    return np.array([2 * x[0], 10 * x[1]])

def hess_f4(x):
    return np.array([[2, 0], [0, 10]])

# Initial seed and damping factor
x0 = np.array([3.3, 3.3])
lambda_factor = 0.1

# Perform optimization and plotting for each function
plt.figure(figsize=(14, 10))
functions = [(f1, grad_f1, hess_f1, "f1: x^2 + y^2"),
             (f2, grad_f2, hess_f2, "f2: 2x^2 + 0.5y^2"),
             (f3, grad_f3, hess_f3, "f3: x^2 + 10y^2"),
             (f4, grad_f4, hess_f4, "f4: x^2 + 5y^2")]

for i, (f, grad_f, hess_f, title) in enumerate(functions, 1):
    plt.subplot(2, 2, i)
    gd_path = gradient_descent_newton(grad_f, x0, alpha=0.1, max_iters=100)
    newton_path = damped_newton(f, grad_f, hess_f, x0, lambda_factor=lambda_factor, max_iters=100)
    plot_contours_and_paths(f, gd_path, newton_path, title)

plt.tight_layout()

plt.show()