import numpy as np
import matplotlib.pyplot as plt

def lasso_loss(X, Y, W, lambda_):
    """Lasso loss function."""
    residual = Y - X @ W
    return 0.5 * np.sum(residual ** 2) + lambda_ * np.sum(np.abs(W))

def subgradient_lasso(X, Y, W, lambda_):
    """Compute subgradient of Lasso loss function."""
    residual = Y - X @ W
    grad = -X.T @ residual
    subgrad = grad + lambda_ * np.sign(W)
    return subgrad

def gradient_descent_lasso(X, Y, lambda_, initial_w, num_iterations=100, constant_step=0.1):
    """Perform gradient descent on the Lasso loss."""
    ws_constant, ws_diminishing = [initial_w.copy()], [initial_w.copy()]
    w_const, w_dim = initial_w.copy(), initial_w.copy()

    for k in range(1, num_iterations + 1):
        # Compute subgradient
        subgrad_const = subgradient_lasso(X, Y, w_const, lambda_)
        subgrad_dim = subgradient_lasso(X, Y, w_dim, lambda_)

        # Constant step size
        w_const = w_const - constant_step * subgrad_const
        ws_constant.append(w_const.copy())

        # Diminishing step size
        diminishing_step = constant_step / k
        w_dim = w_dim - diminishing_step * subgrad_dim
        ws_diminishing.append(w_dim.copy())

    return np.array(ws_constant), np.array(ws_diminishing)


X = np.array([
    [0.5, 1], [1.5, -2], [-1, 1], [-0.5, -1.5],
    [2, 0.5], [0, 1.5], [-1.5, -0.5], [1, -1],
    [1.5, 1], [-2, 0]
])
Y = np.array([1, -1, 0, 0, 1, 2, -2, 0, 1, -1])

# Parameters for the optimization
lambdas = [1, 5]
initial_w = np.array([-2, -2])

# Generate contour plots and optimization paths
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
step_size = 0.1
num_iterations = 50

for idx, lambda_ in enumerate(lambdas):
    # Compute Lasso loss values on a grid for contour plotting
    w1_vals = np.linspace(-3, 3, 100)
    w2_vals = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.array([[lasso_loss(X, Y, np.array([w1, w2]), lambda_) for w1, w2 in zip(w1_row, w2_row)] for w1_row, w2_row in zip(W1, W2)])

    # Perform gradient descent
    ws_constant, ws_diminishing = gradient_descent_lasso(X, Y, lambda_, initial_w, num_iterations=num_iterations, constant_step=step_size)

    # Plot contour map with constant step size path
    ax = axes[idx, 0]
    ax.contour(W1, W2, Z, levels=20, cmap='viridis')
    ax.plot(ws_constant[:, 0], ws_constant[:, 1], 'ro-', markersize=3, label='Optimization Path')
    ax.plot(initial_w[0], initial_w[1], 'bo', label='Start Point')
    ax.set_title(f"Lasso Contour (λ = {lambda_}) with Constant Step Size")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.legend()

    # Plot contour map with diminishing step size path
    ax = axes[idx, 1]
    ax.contour(W1, W2, Z, levels=20, cmap='viridis')
    ax.plot(ws_diminishing[:, 0], ws_diminishing[:, 1], 'ro-', markersize=3, label='Optimization Path')
    ax.plot(initial_w[0], initial_w[1], 'bo', label='Start Point')
    ax.set_title(f"Lasso Contour (λ = {lambda_}) with Diminishing Step Size")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.legend()

plt.tight_layout()
plt.show()