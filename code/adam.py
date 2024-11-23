from init import Rosenbrock
import numpy as np
from init import grad_f
from init import plot_rosenbrock_with_points
import numpy as np
import matplotlib.pyplot as plt


def adam(f, grad_f, x0, epsilon=1e-6, max_iter=1000, alpha=0.001, beta1=0.9, beta2=0.999, epsilon_adam=1e-8):
    """
    Perform Adam optimization in 2D space.
    
    Parameters:
    -----------
    f : function
        Objective function to minimize
    grad_f : function
        Gradient of the objective function
    x0 : tuple
        Initial point (x1, x2)
    epsilon : float
        Convergence criterion for gradient norm
    max_iter : int
        Maximum number of iterations
    alpha : float
        Learning rate
    beta1 : float
        Exponential decay rate for first moment estimates
    beta2 : float
        Exponential decay rate for second moment estimates
    epsilon_adam : float
        Small constant to prevent division by zero
    """
    # Convert initial point to numpy array
    x_k = np.array(x0, dtype=float)
    
    # Initialize first and second moment estimates
    m_k = np.zeros_like(x_k)
    v_k = np.zeros_like(x_k)
    
    # Store visited points for visualization
    visited_points = [x_k.copy()]
    
    k = 0
    while k < max_iter:
        # Compute gradient at current point
        grad = np.array(grad_f(x_k[0], x_k[1]))
        
        # Check for convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < epsilon:
            print(f"Converged in {k} iterations.")
            break
        
        # Update biased first moment estimate
        m_k = beta1 * m_k + (1 - beta1) * grad
        
        # Update biased second moment estimate
        v_k = beta2 * v_k + (1 - beta2) * np.square(grad)
        
        # Compute bias-corrected first moment estimate
        m_k_hat = m_k / (1 - beta1**(k + 1))
        
        # Compute bias-corrected second moment estimate
        v_k_hat = v_k / (1 - beta2**(k + 1))
        
        # Update parameters
        x_k = x_k - alpha * m_k_hat / (np.sqrt(v_k_hat) + epsilon_adam)
        
        # Store current point
        visited_points.append(x_k.copy())
        
        k += 1
    
    if k == max_iter:
        print("Warning: Maximum iterations reached without convergence.")
    
    return x_k, visited_points

# Test the implementation
if __name__ == "__main__":
    # Set initial point
    x0 = (-1.0, -1.0)
    
    # Generate grid for visualization
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = Rosenbrock(X1, X2)
    
    # Run Adam optimization
    opt_point_adam, visited_points_adam = adam(
        Rosenbrock, 
        grad_f, 
        x0, 
        epsilon=1e-6,
        alpha=0.001,  # Try adjusting this if convergence is poor
        max_iter=2000  # Increased from 1000
    )
    
    # Plot results
    plot_rosenbrock_with_points(X1, X2, Z, visited_points_adam, trial_num=1, type="adam")
    
    print(f"Final position: {opt_point_adam}")
    print(f"Final objective value: {Rosenbrock(opt_point_adam[0], opt_point_adam[1])}")