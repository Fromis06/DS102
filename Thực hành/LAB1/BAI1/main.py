import numpy as np
from function import evaluate_function, compute_gradient, compute_hessian
from optimizer import newton_raphson

if __name__ == "__main__":
    initial_point = np.array([0.0, 0.0])
    optimal_point = newton_raphson(
        initial_point=initial_point,
        gradient_fn=compute_gradient,
        hessian_fn=compute_hessian,
        max_iterations=100,
        tolerance=1e-4
    )
    optimal_value = evaluate_function(optimal_point)
    print("Điểm cực trị:", np.round(optimal_point, 4))
    print("Giá trị tại điểm cực trị:", round(optimal_value, 4))