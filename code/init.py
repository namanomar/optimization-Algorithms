import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Defining the rosenbrock function
def Rosenbrock(x1,x2):
  return (1 - x1)**2 + 10*(x2 - x1*x1)**2


def grad_f(x1, x2):
    """
    Compute the gradient of the Rosenbrock function at the point (x1, x2).
    """
    dfdx1 = -2 * (1 - x1) - 40 * x1 * (x2 - x1**2)
    dfdx2 = 20 * (x2 - x1**2)
    return dfdx1, dfdx2

def plot_rosenbrock_with_points(X1, X2, Z, points, trial_num, type, zoom_xlim=None, zoom_ylim=None):
    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # 3D Plot
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='k', alpha=0.4)
    ax3d.view_init(elev=40, azim=30)

    # Extract coordinates for path points
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    z_points = [Rosenbrock(p[0], p[1]) for p in points]
    z_points_above = [z + 5 for z in z_points]

    # Plot optimization path on 3D surface
    ax3d.plot(x_points, y_points, z_points_above, color='red', marker='o', markersize=6, linestyle='-', linewidth=2, label="Optimization Path")

    # Highlight the final point
    final_point = points[-1]
    final_z = Rosenbrock(final_point[0], final_point[1]) + 5
    ax3d.scatter(final_point[0], final_point[1], final_z, color='blue', s=100, label='Final Point', edgecolors='black', zorder=5)

    ax3d.set_xlabel('x1')
    ax3d.set_ylabel('x2')
    ax3d.set_zlabel('Rosenbrock(x1, x2)')
    ax3d.legend()

    # 2D Contour Plot (Zoomed)
    ax2d.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7)
    ax2d.contour(X1, X2, Z, levels=50, colors='black', linewidths=0.5)
    ax2d.plot(x_points, y_points, color='red', marker='o', markersize=4, linestyle='-', linewidth=1.5, label="Optimization Path")
    ax2d.scatter(final_point[0], final_point[1], color='blue', s=60, label='Final Point', edgecolors='black', zorder=5)

    # Set zoomed limits if provided
    if zoom_xlim:
        ax2d.set_xlim(zoom_xlim)
    if zoom_ylim:
        ax2d.set_ylim(zoom_ylim)

    ax2d.set_xlabel('x1')
    ax2d.set_ylabel('x2')
    ax2d.legend()

    # Save and show the plot
    plt.show()


def main():
    # Plotting the rosenbrock function
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = Rosenbrock(X1, X2)

    # Plot the 3D surface
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.view_init(elev=20, azim=135)


    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Rosenbrock(x1, x2)')
    ax.set_title('3D Plot of the Rosenbrock Function')
    plt.show()

main()