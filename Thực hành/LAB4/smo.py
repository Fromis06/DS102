import numpy as np


class LinearSVM:
    """SVM phân loại nhị phân với biên mềm, giải bài toán đối ngẫu bằng SMO."""

    def __init__(self, c: float = 2.0, tolerance: float = 1e-4, max_updates: int = 20_000):
        self.c = c
        self.tolerance = tolerance
        self.max_updates = max_updates
        self.weights = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        kernel = features @ features.T
        diagonal = kernel.diagonal()
        alpha = np.zeros(len(labels))
        scores = np.zeros(len(labels))
        bound_tolerance = min(1e-10, self.c * 1e-4)

        for updates in range(self.max_updates + 1):
            can_increase = np.where(labels == 1, self.c - alpha, alpha)
            can_decrease = np.where(labels == 1, alpha, self.c - alpha)
            residual = labels - scores
            i = np.argmax(np.where(can_increase > bound_tolerance, residual, -np.inf))
            j = np.argmin(np.where(can_decrease > bound_tolerance, residual, np.inf))
            gap = residual[i] - residual[j]
            if gap <= self.tolerance:
                self.weights = (alpha * labels) @ features
                self.bias = (residual[i] + residual[j]) / 2
                self.alpha = alpha
                self.updates = updates
                return self

            if updates == self.max_updates:
                break
            curvature = diagonal[i] + diagonal[j] - 2 * kernel[i, j]
            limit = min(can_increase[i], can_decrease[j])
            step = min(limit, gap / curvature) if curvature > 0 else limit
            alpha[i] += labels[i] * step
            alpha[j] -= labels[j] * step
            scores += step * (kernel[:, i] - kernel[:, j])

        raise RuntimeError(f"SMO chưa hội tụ sau {self.max_updates} lần cập nhật")

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(features) >= 0, 1, -1)
