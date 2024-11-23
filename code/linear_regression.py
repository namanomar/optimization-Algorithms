import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gradient Descent for Linear Regression
def linear_regression_gd(X, Y, alpha, max_iter):
    N, d = X.shape
    X_aug = np.hstack((np.ones((N, 1)), X))
    W = np.random.randn(d + 1, 1)
    
    for k in range(max_iter):
        Y_pred = X_aug @ W
        E = Y_pred - Y
        grad = (1 / N) * (X_aug.T @ E)
        W = W - alpha * grad
    return W, Y_pred

# Plot Actual vs Predicted values
def plot_actual_vs_predicted(Y, Y_pred):
    plt.figure(figsize=(8, 6))
    plt.plot(Y, label='Actual', marker='o')
    plt.plot(Y_pred, label='Predicted', marker='x')
    plt.xlabel("Data Point")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()


# 3D Plane Plot with Predictions (uses only two features)
def plot_3d_plane(X, Y, W):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Select only the first two features for 3D plot
    X_subset = X[:, :2]
    X_aug = np.hstack((np.ones((X_subset.shape[0], 1)), X_subset))
    Y_pred = X_aug @ W[:3]  # Use only weights for bias and first two features

    # Scatter plot of actual data points
    ax.scatter(X_subset[:, 0], X_subset[:, 1], Y, color='blue', label="Actual")

    # Surface plot for predicted plane
    x1_range = np.linspace(min(X_subset[:, 0]), max(X_subset[:, 0]), 10)
    x2_range = np.linspace(min(X_subset[:, 1]), max(X_subset[:, 1]), 10)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    Z = W[0] + W[1] * x1_grid + W[2] * x2_grid
    ax.plot_surface(x1_grid, x2_grid, Z, color='orange', alpha=0.5, rstride=100, cstride=100)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_title("3D Plane Fit")
    ax.legend()
    plt.show()


np.random.seed(0)
X = np.random.randn(10, 10)
Y = np.random.randn(10, 1)
alpha = 0.01
max_iter = 1000


W_final, Y_pred = linear_regression_gd(X, Y, alpha, max_iter)

# Plot Actual vs Predicted values
plot_actual_vs_predicted(Y, Y_pred)

# Plot 3D plane using only the first two features
plot_3d_plane(X, Y, W_final)
