import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Link to google colab: https://colab.research.google.com/drive/1V0zEleb9_UEAl9LSHbb-PK7G-GezlhA8

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

# new thingy-------
def evolutionary_strategy(f, x0, sigma0=1, max_iters=1000, tol=1e-6):
    x = x0
    sigma = sigma0
    history = [x0]

    lr=0.01

    for i in range(max_iters):
        perturbation = np.random.normal(0, sigma, size=x0.shape)
        x_next = x + perturbation

        if f(x_next) < f(x):
            x = x_next
            sigma *= (1 + lr)
        else:
            sigma *= (1 - lr)

        c=0.817
        p = 1

        # regla exito
        if p > 1/5:
          sigma = sigma / c

        elif p < 1/5:
          sigma = sigma * c

        else:
          sigma = sigma


        history.append(x)

        if np.linalg.norm(perturbation) < tol:
            break

    return x, np.array(history)



A = 10

# Sphere function
def sphere(x):
    return np.sum(x**2)

def sphere_grad(x):
    return 2*x

def sphere_hess(x):
    return 2*np.eye(len(x))

# Rastrigin function
def rastrigin(x):
    n = len(x)
    return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))

def rastrigin_grad(x):
    return 2*x + 2*A*np.pi*np.sin(2*np.pi*x)

def rastrigin_hess(x):
    return 2*np.eye(len(x)) + 4*(A*np.pi**2)*np.diag(np.cos(2*np.pi*x))

# Drop-Wave function
def drop_wave(x):
    return -(1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))) / (0.5 * (x[0]**2 + x[1]**2) + 2)

def drop_wave_grad(x):
    x1, x2 = x
    denom = 0.5 * (x1**2 + x2**2) + 2
    numer = np.cos(12 * np.sqrt(x1**2 + x2**2))

    dfdx1 = ((x1 * (24 * numer + 1)) * np.sin(12 * np.sqrt(x1**2 + x2**2))
             - (x1 * (1 + numer))) / (denom ** 2)
    dfdx2 = ((x2 * (24 * numer + 1)) * np.sin(12 * np.sqrt(x1**2 + x2**2))
             - (x2 * (1 + numer))) / (denom ** 2)

    return np.array([dfdx1, dfdx2])

def drop_wave_hess(x):
    # Placeholder
    return np.eye(2)

# Matyas function
def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def matyas_grad(x):
    return np.array([0.52*x[0] - 0.48*x[1], 0.52*x[1] - 0.48*x[0]])

def matyas_hess(x):
    return np.array([[0.52, -0.48], [-0.48, 0.52]])

# Rosenbrock function
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

def rosenbrock_hess(x):
    return np.array([[2 - 400*x[1] + 1200*x[0]**2, -400*x[0]], [-400*x[0], 200]])

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

    plt.title(f.__name__)
    plt.show()

# Tabulate sequences
def tabulate_history(history):
    df = pd.DataFrame(history, columns=["x", "y"])
    return df

# Tests
x0 = np.array([1.0, 1.0])

functions = [(sphere, sphere_grad, sphere_hess),
             (rastrigin, rastrigin_grad, rastrigin_hess),
             (drop_wave, drop_wave_grad, drop_wave_hess),
             (matyas, matyas_grad, matyas_hess),
             (rosenbrock, rosenbrock_grad, rosenbrock_hess)]

for func, grad, hess in functions:
    x_min_gd, history_gd = gradient_descent(func, grad, x0)
    print(f"Minimum of {func.__name__} with Gradient Descent:", x_min_gd)

    x_min_newton, history_newton = newton_method(func, grad, hess, x0)
    print(f"Minimum of {func.__name__} with Newton's method:", x_min_newton)

    x_min_es, history_es = evolutionary_strategy(func, x0)
    print(f"Minimum of {func.__name__} with Evolutionary Strategy:", x_min_es)

    plot_contour(func, history=history_gd)
    print(f"History of {func.__name__} with Gradient Descent:")
    print(tabulate_history(history_gd))

    plot_contour(func, history=history_newton)
    print(f"History of {func.__name__} with Newton's method:")
    print(tabulate_history(history_newton))

    plot_contour(func, history=history_es)
    print(f"History of {func.__name__} with evolutionary strategy:")
    print(tabulate_history(history_es))
