import numpy as np
import matplotlib.pyplot as plt

def logistic_regression_SGD(X, Y, alpha=0.01, max_it=1000, batch_size=1):
    N, d = X.shape
    X = np.hstack([np.ones((N, 1)), X])  # Augment X to include bias term
    
    # Initialize weights (W) to zeros
    W = np.zeros(d + 1)  # Including bias term
    
    # Store loss values for plotting
    loss_history = []
    
    # Number of batches
    n_batches = max(N // batch_size, 1)
    
    # SGD
    for k in range(max_it):
        # Shuffle the data
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        # Process mini-batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, N)
            
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]
            
            # Logistic regression hypothesis (sigmoid function)
            h = 1 / (1 + np.exp(-np.dot(X_batch, W)))
            
            # Compute the gradient of the loss function for the batch
            gradient = (1 / len(X_batch)) * np.dot(X_batch.T, (h - Y_batch))
            
            # Update the weights using the batch gradient
            W -= alpha * gradient
        
        # Compute and store the full loss every 100 iterations
        if k % 100 == 0:
            h_full = 1 / (1 + np.exp(-np.dot(X, W)))
            loss = -np.mean(Y * np.log(h_full + 1e-15) + (1 - Y) * np.log(1 - h_full + 1e-15))
            loss_history.append(loss)
            print(f"Iteration {k}, Loss: {loss}")
    
    return W, loss_history



# Generate  data
np.random.seed(42)
N = 100  # Number of data points
d = 2    # Number of features


# Create random feature matrix X (N x d)
X = np.random.randn(N, d)

# Generate random binary labels (0 or 1)
Y = np.random.randint(0, 2, size=N)

# Apply SGD for logistic regression
# Using a larger learning rate since we're doing SGD
W_final, loss_history = logistic_regression_SGD(X, Y, alpha=0.1,max_it=1000,batch_size=1)  # batch_size=1 for pure SGD
print("Final optimized weights:", W_final)

# Create figure for plotting
plt.figure(figsize=(15, 5))

# First plot (loss curve)
plt.subplot(1, 3, 1)
plt.plot(range(0, 1000, 100), loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve during SGD')
plt.grid(True)

# Second plot (decision boundary)
plt.subplot(1, 3, 2)

# Generate a grid of points for decision boundary visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Prepare grid points for prediction
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])

# Compute predictions for the grid points
Z = 1 / (1 + np.exp(-np.dot(grid_points_bias, W_final)))
Z = Z.reshape(xx.shape)

# Plot decision boundary and data points
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ffcccc', '#cce5ff'], alpha=0.7)
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary (SGD)')
plt.legend()
plt.grid(True)

# Third plot (trajectory visualization)
plt.subplot(1, 3, 3)
# Generate predictions for the entire dataset at regular intervals
traj_points = np.linspace(0, len(loss_history)-1, 10).astype(int)
for i, point in enumerate(traj_points):
    plt.contour(xx, yy, Z, levels=[0.5], 
                colors=[plt.cm.viridis(i/len(traj_points))],
                alpha=0.5)
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()