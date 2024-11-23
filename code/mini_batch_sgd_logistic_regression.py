import numpy as np
import matplotlib.pyplot as plt

def logistic_regression_BatchSGD(X, Y, batch_size, alpha=0.01, max_it=1000):
    N, d = X.shape
    X = np.hstack([np.ones((N, 1)), X])  # Augment X to include bias term
    
    W = np.zeros(d + 1)  
    loss_history = []
    batch_losses = []  
    n_batches = N // batch_size
    if N % batch_size != 0:
        n_batches += 1
    
    # Batch SGD
    for k in range(max_it):
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        epoch_batch_losses = []  # Store losses for each batch in this epoch
        
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
            
            # Compute batch loss
            batch_loss = -np.mean(Y_batch * np.log(h + 1e-15) + 
                                (1 - Y_batch) * np.log(1 - h + 1e-15))
            epoch_batch_losses.append(batch_loss)
        
        # Store average batch loss for this epoch
        batch_losses.append(np.mean(epoch_batch_losses))
        
        # Compute and store the full loss every 100 iterations
        if k % 100 == 0:
            h_full = 1 / (1 + np.exp(-np.dot(X, W)))
            loss = -np.mean(Y * np.log(h_full + 1e-15) + 
                          (1 - Y) * np.log(1 - h_full + 1e-15))
            loss_history.append(loss)
            print(f"Iteration {k}, Loss: {loss}")
    
    return W, loss_history, batch_losses

# Generate  data
np.random.seed(42)
N = 1000  # Increased number of data points for better batch size comparison
d = 2     # Number of features

# Create random feature matrix X (N x d)
X = np.random.randn(N, d)

# Generate more separable binary labels using a true linear decision boundary
true_w = np.array([1, -2, 2])
X_augmented = np.hstack([np.ones((N, 1)), X])
probabilities = 1 / (1 + np.exp(-np.dot(X_augmented, true_w)))
Y = (probabilities > 0.5).astype(int)

# Try different batch sizes
batch_sizes = [1, 10, 50, 100]
results = {}

for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    W_final, loss_history, batch_losses = logistic_regression_BatchSGD(
        X, Y, 
        batch_size=batch_size,
        alpha=0.1,
        max_it=1000
    )
    results[batch_size] = (W_final, loss_history, batch_losses)

# Visualization
plt.figure(figsize=(20, 10))

# Plot 1: Loss curves comparison
plt.subplot(2, 2, 1)
for batch_size in batch_sizes:
    plt.plot(range(0, 1000, 100), 
            results[batch_size][1], 
            label=f'Batch Size = {batch_size}')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Batch Sizes')
plt.legend()
plt.grid(True)

# Plot 2: Batch Loss Volatility
plt.subplot(2, 2, 2)
for batch_size in batch_sizes:
    plt.plot(results[batch_size][2][:100],  # Plot first 100 batch losses
            label=f'Batch Size = {batch_size}')
plt.xlabel('Batch Updates')
plt.ylabel('Batch Loss')
plt.title('Batch Loss Volatility (First 100 Updates)')
plt.legend()
plt.grid(True)

# Plot 3: Decision Boundaries Comparison
plt.subplot(2, 2, 3)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])

for i, batch_size in enumerate(batch_sizes):
    W = results[batch_size][0]
    Z = 1 / (1 + np.exp(-np.dot(grid_points_bias, W)))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], 
                colors=[plt.cm.viridis(i/len(batch_sizes))],
                label=f'Batch Size = {batch_size}')

plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='red', 
           label='Class 0', alpha=0.5, s=20)
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', 
           label='Class 1', alpha=0.5, s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries for Different Batch Sizes')
plt.legend()
plt.grid(True)

# Plot 4: Convergence Speed Comparison
plt.subplot(2, 2, 4)
convergence_threshold = 0.1
epochs_to_converge = {}

for batch_size in batch_sizes:
    losses = results[batch_size][2]
    for i, loss in enumerate(losses):
        if loss < convergence_threshold:
            epochs_to_converge[batch_size] = i
            break
    else:
        epochs_to_converge[batch_size] = len(losses)

plt.bar(range(len(batch_sizes)), 
        [epochs_to_converge[bs] for bs in batch_sizes],
        tick_label=[f'Batch Size\n{bs}' for bs in batch_sizes])
plt.ylabel('Updates Until Convergence')
plt.title('Convergence Speed Comparison')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()