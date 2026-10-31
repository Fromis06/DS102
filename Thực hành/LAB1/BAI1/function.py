import numpy as np

def evaluate_function(point):
    x, y = point
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def compute_gradient(point):
    x, y = point
    df_dx = 10*x + 8*y - 34
    df_dy = 8*x + 10*y - 38
    return np.array([df_dx, df_dy])

def compute_hessian(point):
    return np.array([
        [10.0, 8.0],
        [8.0, 10.0]
    ])