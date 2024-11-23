import numpy as np
import matplotlib.pyplot as plt

def logistic_regression_GD(X, Y, alpha=0.01, max_it=1000):
    N, d = X.shape
    X = np.hstack([np.ones((N, 1)), X])  
    W = np.zeros(d + 1)  
    loss_history = []
    
    # Gradient Descent
    for k in range(max_it):
        h = 1 / (1 + np.exp(-np.dot(X, W)))
        
        gradient = (1 / N) * np.dot(X.T, (h - Y)) 
        W -= alpha * gradient  
        if k % 100 == 0:
            loss = -np.mean(Y * np.log(h + 1e-15) + (1 - Y) * np.log(1 - h + 1e-15))  
            loss_history.append(loss)
            print(f"Iteration {k}, Loss: {loss}")
    
    return W, loss_history


np.random.seed(42)
N = 100  # Number of data points
d = 2    # Number of features

X = np.random.randn(N, d)

Y = np.random.randint(0, 2, size=N) 
W_final, loss_history = logistic_regression_GD(X, Y, alpha=0.1, max_it=500)
print("Final optimized weights:", W_final)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(0, 500, 100), loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve during Gradient Descent')
plt.grid(True)

plt.subplot(1, 2, 2)

# Generate a grid of points for decision boundary visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))


grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])

Z = 1 / (1 + np.exp(-np.dot(grid_points_bias, W_final)))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ffcccc', '#cce5ff'], alpha=0.7)
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary of Logistic Regression')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()