import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 100
X = np.random.uniform(-2, 2, (n_samples, 2))
true_w = np.array([1, -0.5])
y = np.dot(X, true_w) + np.random.normal(0, 0.1, n_samples)

# Create mesh grid for contour plot
x1_range = np.linspace(-2.5, 2.5, 100)
x2_range = np.linspace(-2.5, 2.5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = np.zeros_like(X1)

# Calculate loss surface
for i in range(len(x1_range)):
    for j in range(len(x2_range)):
        w = np.array([X1[i,j], X2[i,j]])
        predictions = np.dot(X, w)
        Z[i,j] = np.mean((predictions - y) ** 2)

# Set same initial point for all methods
initial_w = np.array([-2.0, 2.0])

# Optimization functions
def gradient_descent(X, y, w_init, learning_rate=0.1, n_iterations=100):
    w = w_init.copy()
    path = [w.copy()]
    
    for _ in range(n_iterations):
        gradient = 2 * X.T.dot(X.dot(w) - y) / len(y)
        w -= learning_rate * gradient
        path.append(w.copy())
    
    return np.array(path)

def sgd(X, y, w_init, learning_rate=0.1, n_iterations=100):
    w = w_init.copy()
    path = [w.copy()]
    
    for _ in range(n_iterations):
        idx = np.random.randint(len(y))
        gradient = 2 * X[idx].T.dot(X[idx].dot(w) - y[idx])
        w -= learning_rate * gradient
        path.append(w.copy())
    
    return np.array(path)

def mini_batch_sgd(X, y, w_init, batch_size=32, learning_rate=0.1, n_iterations=100):
    w = w_init.copy()
    path = [w.copy()]
    
    for _ in range(n_iterations):
        indices = np.random.choice(len(y), batch_size)
        X_batch, y_batch = X[indices], y[indices]
        gradient = 2 * X_batch.T.dot(X_batch.dot(w) - y_batch) / batch_size
        w -= learning_rate * gradient
        path.append(w.copy())
    
    return np.array(path)

# Run optimizations
learning_rate = 0.1
n_iterations = 100
batch_size = 32

path_gd = gradient_descent(X, y, initial_w, learning_rate, n_iterations)
path_sgd = sgd(X, y, initial_w, learning_rate, n_iterations)
path_mbgd = mini_batch_sgd(X, y, initial_w, batch_size, learning_rate, n_iterations)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Contour plot with optimization paths
plt.subplot(221)
plt.contour(X1, X2, Z, levels=np.logspace(-2, 2, 20))
plt.plot(path_gd[:,0], path_gd[:,1], 'b.-', label='GD', alpha=0.7, markersize=3)
plt.plot(path_sgd[:,0], path_sgd[:,1], 'r.-', label='SGD', alpha=0.7, markersize=3)
plt.plot(path_mbgd[:,0], path_mbgd[:,1], 'g.-', label='BatchGD', alpha=0.7, markersize=3)
plt.plot(initial_w[0], initial_w[1], 'ko', label='Start', markersize=10)
plt.plot(true_w[0], true_w[1], 'k*', label='True weights', markersize=10)
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Optimization Paths in Parameter Space')
plt.legend()
plt.grid(True)

# Plot 2: Loss convergence
plt.subplot(222)
loss_gd = [np.mean((np.dot(X, w) - y) ** 2) for w in path_gd]
loss_sgd = [np.mean((np.dot(X, w) - y) ** 2) for w in path_sgd]
loss_mbgd = [np.mean((np.dot(X, w) - y) ** 2) for w in path_mbgd]

plt.plot(loss_gd, 'b-', label='GD')
plt.plot(loss_sgd, 'r-', label='SGD')
plt.plot(loss_mbgd, 'g-', label='BatchGD')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Convergence')
plt.legend()
plt.grid(True)
plt.yscale('log')

# Plot 3: 3D surface plot
ax = plt.subplot(212, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)
ax.plot(path_gd[:,0], path_gd[:,1], [np.mean((np.dot(X, w) - y) ** 2) for w in path_gd], 'b.-', label='GD', alpha=0.7)
ax.plot(path_sgd[:,0], path_sgd[:,1], [np.mean((np.dot(X, w) - y) ** 2) for w in path_sgd], 'r.-', label='SGD', alpha=0.7)
ax.plot(path_mbgd[:,0], path_mbgd[:,1], [np.mean((np.dot(X, w) - y) ** 2) for w in path_mbgd], 'g.-', label='BatchGD', alpha=0.7)
ax.scatter(initial_w[0], initial_w[1], np.mean((np.dot(X, initial_w) - y) ** 2), color='k', s=100, label='Start')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface and Optimization Paths')
ax.legend()
plt.tight_layout()
plt.savefig("./output/comparision_linear_regression")
plt.show()