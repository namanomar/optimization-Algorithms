import numpy as np
import matplotlib.pyplot as plt

def func1(x1, x2):
    return x1 + 2 * np.abs(x2)

def approximate_subgradient(f, x, epsilon=1e-6):
    """
    Compute an approximate subgradient of a function f at point x.
    """
    subgrad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.array(x, dtype=float)
        x_minus = np.array(x, dtype=float)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        subgrad[i] = (f(x_plus[0], x_plus[1]) - f(x_minus[0], x_minus[1])) / (2 * epsilon)
    return subgrad

x1_vals = np.linspace(-2, 2, 20)
x2_vals = np.linspace(-2, 2, 20)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = func1(X1, X2)

# Plotting the function and subgradients
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotting the subgradients as arrows
for x1 in x1_vals:
    for x2 in x2_vals:
        g = approximate_subgradient(lambda x1, x2: func1(x1, x2), [x1, x2])
        z = func1(x1, x2)
        ax.quiver(x1, x2, z, g[0], g[1], 0, length=0.1, color='red')

# Set labels and title
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
ax.set_title("3D plot of f(x1, x2) with subgradients")
import os
print(os.getcwd())

plt.show()
