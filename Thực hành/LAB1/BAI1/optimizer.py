import numpy as np

def newton_raphson(initial_point, gradient_fn, hessian_fn, max_iterations=100, tolerance=1e-4):
    current_point = initial_point.astype(float)
    for i in range(max_iterations):
        gradient = gradient_fn(current_point)
        if np.linalg.norm(gradient) < tolerance:
            return current_point
        hessian = hessian_fn(current_point)
        step = np.linalg.solve(hessian, -gradient)
        current_point = current_point + step

    return current_point