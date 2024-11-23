# Optimization Algorithms in Python

This project implements **unconstrained optimization algorithms** from scratch using Python. The goal is to provide an intuitive understanding of these algorithms through detailed implementation and explanations.

---

## 🚀 Features

- All algorithms implemented from scratch.
- Step-by-step explanation for each algorithm.
- Easy-to-read code with detailed comments.
- Suitable for beginners and advanced learners.

---

## 📚 Included Algorithms

### Gradient-Based Optimization
- **Gradient Descent**  
  Iteratively updates parameters in the direction of the negative gradient to minimize the objective function.

- **Momentum Gradient Descent**  
  Accelerates convergence using momentum to escape local minima and optimize the search direction.

- **Nesterov Accelerated Gradient**  
  Improves upon Momentum Gradient Descent by adjusting gradients based on a lookahead point.

and more ...

### Second-Order Optimization
- **Newton's Method**  
  Leverages second-order derivatives (Hessian) for faster convergence towards local minima.

- **Quasi-Newton Methods**  
  Approximates the Hessian for optimization without computing it explicitly.


---

## 📁 Project Structure


-/<br>
--- code/ <br>
 --- comparision/ <br>
 --- notebook/ <br>
 --- output/ <br>
 --- requirements.txt <br>
 --- setup.txt <br>



## 🔧 Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/namanomar/optimization-algorithms.git
   cd optimization-algorithms

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Run examples:

    ```
    python code/linear_regression.py
    ```

📊 Visualization
Plots are used to visualize:

- Convergence over iterations.
- Contour plots for 2D functions.
- Learning rate effect on optimization performance.

