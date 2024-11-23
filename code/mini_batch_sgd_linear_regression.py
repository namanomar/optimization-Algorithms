import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Mini-batch SGD for Linear Regression
def linear_regression_minibatch_sgd(X, Y, alpha, max_iter, minibatch_size):
    N, d = X.shape
    X_aug = np.hstack((np.ones((N, 1)), X))  
    W = np.random.randn(d + 1, 1)  # Initial weights with bias

    for k in range(max_iter):
        # Shuffle the dataset and create mini-batches
        indices = np.random.permutation(N)
        for i in range(0, N, minibatch_size):
            minibatch_indices = indices[i:i+minibatch_size]
            X_mini = X_aug[minibatch_indices]
            Y_mini = Y[minibatch_indices]
            
            # Compute predictions for the selected mini-batch
            Y_pred_mini = X_mini @ W
            error_mini = Y_pred_mini - Y_mini
            
            # Compute gradient for the mini-batch
            grad_mini = X_mini.T @ error_mini / minibatch_size
            
            # Update weights using the gradient and learning rate
            W = W - alpha * grad_mini

    # Final prediction using all data points
    Y_pred = X_aug @ W
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

    X_subset = X[:, :2]
    X_aug = np.hstack((np.ones((X_subset.shape[0], 1)), X_subset))
    Y_pred = X_aug @ W[:3]  
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
X = np.random.randn(10, 10)  # 10 samples, 10 features
Y = np.random.randn(10, 1)  # 10 target values
alpha = 0.01  # Learning rate
max_iter = 1000  # Number of iterations
minibatch_size = 5  # Mini-batch size

# Train model with Mini-batch SGD
W_final, Y_pred = linear_regression_minibatch_sgd(X, Y, alpha, max_iter, minibatch_size)

# Plot Actual vs Predicted values
plot_actual_vs_predicted(Y, Y_pred)

# Plot 3D plane using only the first two features
plot_3d_plane(X, Y, W_final)
