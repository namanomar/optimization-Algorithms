import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Quadratic function f(x) = x^T A x + b^T x + c
def f(x):
    A = np.array([[1.0, 0.5],
                  [0.5, 2.0]])
    b = np.array([-1.0, -2.0])
    c = 1.0
    x = np.array(x).reshape(-1, 1)  # Ensure x is a column vector
    return float(0.5 * x.T @ A @ x + b.T @ x + c)

# Gradient of the quadratic function
def grad_f(x):
    A = np.array([[1.0, 0.5],
                  [0.5, 2.0]])
    b = np.array([-1.0, -2.0])
    x = np.array(x)
    return A @ x + b

# Hessian of the quadratic function
def hess_f(x):
    return np.array([[1.0, 0.5],
                    [0.5, 2.0]])

# Gradient Descent with Newton (for comparison)
def gradient_descent_newton(grad_f, x0, alpha=0.1, epsilon=1e-6, max_iters=100):
    x = np.array(x0)
    path = [x.copy()]
    for _ in range(max_iters):
        g = grad_f(x)
        if norm(g) < epsilon:
            break
        x = x - alpha * g
        path.append(x.copy())
    return np.array(path)

# Damped Newton method (using the Hessian)
def damped_newton(f, grad_f, hess_f, x0, alpha=0.1, epsilon=1e-6, max_iters=100):
    x = np.array(x0)
    path = [x.copy()]
    for _ in range(max_iters):
        g = grad_f(x)
        H = hess_f(x)
        if norm(g) < epsilon:
            break
        # Damped Newton step: x = x - alpha * (H^-1 * g)
        x = x - alpha * np.linalg.inv(H) @ g
        path.append(x.copy())
    return np.array(path)

# Conjugate Gradient method
def conjugate_gradient(grad_f, x0, epsilon=1e-6, max_iters=100):
    x = np.array(x0)
    r = -grad_f(x)
    d = r.copy()
    path = [x.copy()]

    for _ in range(max_iters):
        if norm(r) < epsilon:
            break

        # Compute Hessian-vector product (approximation)
        Hd = grad_f(x + 1e-7 * d) - grad_f(x)
        Hd = Hd / 1e-7

        alpha = (r @ r) / (d @ Hd)
        x = x + alpha * d
        r_new = r - alpha * Hd
        beta = (r_new @ r_new) / (r @ r)
        d = r_new + beta * d
        r = r_new

        path.append(x.copy())

    return np.array(path)

# Function to compare optimization methods
def compare_optimization_methods(f, grad_f, hess_f, x0):
    # Run all optimization methods
    gd_path = gradient_descent_newton(grad_f, x0, alpha=0.1)
    newton_path = damped_newton(f, grad_f, hess_f, x0, alpha=0.1)
    cg_path = conjugate_gradient(grad_f, x0)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot settings
    xlims = (-3, 3)
    ylims = (-3, 3)
    x = np.linspace(xlims[0], xlims[1], 100)
    y = np.linspace(ylims[0], ylims[1], 100)
    X, Y = np.meshgrid(x, y)

    # Vectorize the function properly
    Z = np.array([[f([xi, yi]) for xi in x] for yi in y])

    # Plot for Gradient Descent
    axes[0].contour(X, Y, Z, levels=20, cmap='viridis')
    axes[0].plot(gd_path[:, 0], gd_path[:, 1], 'r.-', label="Gradient Descent")
    axes[0].set_title("Gradient Descent")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[0].legend()

    # Plot for Damped Newton
    axes[1].contour(X, Y, Z, levels=20, cmap='viridis')
    axes[1].plot(newton_path[:, 0], newton_path[:, 1], 'g.-', label="Damped Newton")
    axes[1].set_title("Damped Newton")
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₂")
    axes[1].legend()

    # Plot for Conjugate Gradient
    axes[2].contour(X, Y, Z, levels=20, cmap='viridis')
    axes[2].plot(cg_path[:, 0], cg_path[:, 1], 'b.-', label="Conjugate Gradient")
    axes[2].set_title("Conjugate Gradient")
    axes[2].set_xlabel("x₁")
    axes[2].set_ylabel("x₂")
    axes[2].legend()

    plt.savefig("./output/comparision_hessian_methods.png")
    plt.tight_layout()
    plt.show()

    # Print iteration counts
    print(f"Gradient Descent iterations: {len(gd_path)}")
    print(f"Damped Newton iterations: {len(newton_path)}")
    print(f"Conjugate Gradient iterations: {len(cg_path)}")

# Run the comparison
x0 = np.array([2.0, 2.0])
compare_optimization_methods(f, grad_f, hess_f, x0)
