# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Gradient Descent
def gradient_descent(f, gradf, x0, lr=0.01, max_iters=1000, tol=1e-6):
    x = x0
    history = [x0]
    for i in range(max_iters):
        gradient = gradf(x)
        x_next = x - lr * gradient
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next
        history.append(x)
    return x, np.array(history)

# Newtons method
def newton_method(f, gradf, hessf, x0, max_iters=1000, tol=1e-6):
    x = x0
    history = [x0]
    for i in range(max_iters):
        gradient = gradf(x)
        hessian_inv = np.linalg.inv(hessf(x))
        x_next = x - np.dot(hessian_inv, gradient)
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next
        history.append(x)
    return x, np.array(history)

# Test functions
A = 10

def sphere(x):
    return np.sum(x**2)

def sphere_grad(x):
    return 2*x

def sphere_hess(x):
    return 2*np.eye(len(x))

def rastrigin(x):
    n = len(x)
    return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))

def rastrigin_grad(x):
    return 2*x + 2*A*np.pi*np.sin(2*np.pi*x)

def rastrigin_hess(x):
    return 2*np.eye(len(x)) + 4*(A*np.pi**2)*np.diag(np.cos(2*np.pi*x))



# Tests
x0 = np.array([1.0, 1.0])

x_min_gd, history_gd = gradient_descent(sphere, sphere_grad, x0)
print("Minimum with Gradient Descent:", x_min_gd)

x_min_newton, history_newton = newton_method(sphere, sphere_grad, sphere_hess, x0)
print("Minimum with Newton's method:", x_min_newton)

# Plot
def plot_contour(f, x_range=(-2, 2), y_range=(-2, 2), history=None):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar()
    
    if history is not None:
        plt.scatter(history[:, 0], history[:, 1], c='red', marker='o')
        plt.plot(history[:, 0], history[:, 1], c='red', linewidth=2)

    plt.show()

# Plot for Sphere function with Gradient Descent history
plot_contour(sphere, history=history_gd)

# 5. Tabulate sequences:
def tabulate_history(history):
    df = pd.DataFrame(history, columns=["x", "y"])
    return df

# Displaying the history for Sphere function with Gradient Descent:
print(tabulate_history(history_gd))
