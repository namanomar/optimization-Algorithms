import numpy as np
import matplotlib.pyplot as plt
from time import time

class LogisticRegressionComparison:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y, weights):
        h = self.sigmoid(np.dot(X, weights))
        return -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    
    def gradient_descent(self, X, y):
        """Standard Batch Gradient Descent"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        loss_history = []
        time_history = []
        start_time = time()
        
        for i in range(self.max_iterations):
            # Compute predictions
            h = self.sigmoid(np.dot(X, weights))
            
            # Compute gradients
            gradients = (1/n_samples) * np.dot(X.T, (h - y))
            
            # Update weights
            weights -= self.learning_rate * gradients
            
            # Record loss and time
            if i % 10 == 0:
                loss = self.compute_loss(X, y, weights)
                loss_history.append(loss)
                time_history.append(time() - start_time)
                
        return weights, loss_history, time_history
    
    def stochastic_gradient_descent(self, X, y):
        """Stochastic Gradient Descent"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        loss_history = []
        time_history = []
        start_time = time()
        
        for i in range(self.max_iterations):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            
            # Update weights for each sample
            for j in range(n_samples):
                h = self.sigmoid(np.dot(X_shuffled[j:j+1], weights))
                gradients = np.dot(X_shuffled[j:j+1].T, (h - y_shuffled[j:j+1]))
                weights -= self.learning_rate * gradients
            
            # Record loss and time
            if i % 10 == 0:
                loss = self.compute_loss(X, y, weights)
                loss_history.append(loss)
                time_history.append(time() - start_time)
                
        return weights, loss_history, time_history
    
    def mini_batch_gradient_descent(self, X, y, batch_size=32):
        """Mini-batch Gradient Descent"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        loss_history = []
        time_history = []
        start_time = time()
        
        for i in range(self.max_iterations):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            
            # Process mini-batches
            for j in range(0, n_samples, batch_size):
                X_batch = X_shuffled[j:j+batch_size]
                y_batch = y_shuffled[j:j+batch_size]
                
                h = self.sigmoid(np.dot(X_batch, weights))
                gradients = (1/len(X_batch)) * np.dot(X_batch.T, (h - y_batch))
                weights -= self.learning_rate * gradients
            
            # Record loss and time
            if i % 10 == 0:
                loss = self.compute_loss(X, y, weights)
                loss_history.append(loss)
                time_history.append(time() - start_time)
                
        return weights, loss_history, time_history

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
n_features = 2

# Generate more separable data
X = np.random.randn(n_samples, n_features)
true_weights = np.array([1, -2])
y = (np.dot(X, true_weights) + np.random.randn(n_samples) * 0.1 > 0).astype(int)

# Add bias term
X = np.hstack([np.ones((n_samples, 1)), X])

# Initialize and train models
lr_comparison = LogisticRegressionComparison(learning_rate=0.01, max_iterations=200)

# Train all three variants
print("Training  Gradient Descent...")
bgd_weights, bgd_loss, bgd_time = lr_comparison.gradient_descent(X, y)

print("Training Stochastic Gradient Descent...")
sgd_weights, sgd_loss, sgd_time = lr_comparison.stochastic_gradient_descent(X, y)

print("Training Mini-batch Gradient Descent...")
mbgd_weights, mbgd_loss, mbgd_time = lr_comparison.mini_batch_gradient_descent(X, y, batch_size=32)

# Visualization
plt.figure(figsize=(20, 12))

# 1. Loss Convergence Comparison
plt.subplot(2, 2, 1)
plt.plot(bgd_loss, label='Batch GD')
plt.plot(sgd_loss, label='Stochastic GD')
plt.plot(mbgd_loss, label='Mini-batch GD')
plt.xlabel('Iterations (x10)')
plt.ylabel('Loss')
plt.title('Loss Convergence Comparison')
plt.legend()
plt.grid(True)

# 2. Training Time Comparison
plt.subplot(2, 2, 2)
plt.plot(bgd_time, bgd_loss, label='Batch GD')
plt.plot(sgd_time, sgd_loss, label='Stochastic GD')
plt.plot(mbgd_time, mbgd_loss, label='Mini-batch GD')
plt.xlabel('Time (seconds)')
plt.ylabel('Loss')
plt.title('Training Time vs Loss')
plt.legend()
plt.grid(True)

# 3. Decision Boundaries Comparison
plt.subplot(2, 2, 3)
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[np.ones((10000, 1)), xx.ravel(), yy.ravel()]

# Plot decision boundaries
for weights, style in zip([bgd_weights, sgd_weights, mbgd_weights], 
                         ['b-', 'r--', 'g:']):
    Z = lr_comparison.sigmoid(np.dot(grid, weights))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors=style[0], 
                linestyles=[style[1]], label=f'Decision boundary')

plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], label='Class 0', alpha=0.5)
plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], label='Class 1', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries Comparison')
plt.legend(['BGD', 'SGD', 'MBGD', 'Class 0', 'Class 1'])
plt.grid(True)

# 4. Performance Metrics
plt.subplot(2, 2, 4)
final_losses = [bgd_loss[-1], sgd_loss[-1], mbgd_loss[-1]]
final_times = [bgd_time[-1], sgd_time[-1], mbgd_time[-1]]

x = np.arange(3)
width = 0.35

plt.bar(x - width/2, final_losses, width, label='Final Loss')
plt.bar(x + width/2, final_times, width, label='Training Time')
plt.xticks(x, ['GD', 'SGD', 'MBGD'])
plt.ylabel('Value')
plt.title('Final Performance Comparison')
plt.legend()

plt.tight_layout()
plt.savefig("./output/comparision_logistic_regression.png")
plt.show()

# Print numerical comparison
print("\nNumerical Comparison:")
print(f"{'Method':<15} {'Final Loss':>12} {'Training Time':>15}")
print("-" * 45)
print(f"{' GD':<15} {bgd_loss[-1]:>12.6f} {bgd_time[-1]:>15.6f}")
print(f"{'Stochastic GD':<15} {sgd_loss[-1]:>12.6f} {sgd_time[-1]:>15.6f}")
print(f"{'Mini-batch GD':<15} {mbgd_loss[-1]:>12.6f} {mbgd_time[-1]:>15.6f}")