import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def f(x):
    """Quadratic function f(x) = x^T A x + b^T x + c"""
    A = np.array([[1.0, 0.5],
                  [0.5, 2.0]])
    b = np.array([-1.0, -2.0])
    c = 1.0
    x = np.array(x).reshape(-1, 1)  # Ensure x is a column vector
    return float(0.5 * x.T @ A @ x + b.T @ x + c)

def grad_f(x):
    """Gradient of the quadratic function"""
    A = np.array([[1.0, 0.5],
                  [0.5, 2.0]])
    b = np.array([-1.0, -2.0])
    x = np.array(x)  # Ensure x is a numpy array
    return A @ x + b

def hess_f(x):
    """Hessian of the quadratic function"""
    return np.array([[1.0, 0.5],
                    [0.5, 2.0]])

# Conjugate Gradient Method
def conjugate_gradient(grad_f, x0, epsilon=1e-6, max_iters=100):
    x = np.array(x0)
    r = -grad_f(x)
    d = r.copy()
    path = [x.copy()]

    for _ in range(max_iters):
        if norm(r) < epsilon:
            break

        # Compute Hessian-vector product
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

# Initial point
x0 = np.array([2.0, 1.0])

# Run Conjugate Gradient
path = conjugate_gradient(grad_f, x0)

# Plotting the path of optimization on the function's contour
x1_vals = np.linspace(-3, 3, 400)
x2_vals = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.array([[f([x1, x2]) for x1, x2 in zip(x1_vals, row)] for row in X2])

plt.figure(figsize=(8, 6))
cp = plt.contour(X1, X2, Z, levels=30, cmap='coolwarm')
plt.colorbar(cp)

# Plot the path taken by the conjugate gradient method
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], marker='o', color='r', label='Conjugate Gradient Path')

plt.title("Conjugate Gradient Optimization Path")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.grid(True)
plt.show()
