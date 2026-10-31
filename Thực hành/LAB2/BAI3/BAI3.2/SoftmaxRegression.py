import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnSoftmaxRegression

class SoftmaxRegression:
    def __init__(self, learning_rate: float = 0.5, epochs: int = 300, l2_penalty: float = 0.01, num_classes: int = 10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.num_classes = num_classes
        self.weights = None
        self.bias = None
        c_val = 1.0 / self.l2_penalty if self.l2_penalty > 0 else 1e10

        self.model = SklearnSoftmaxRegression(
            penalty="l2",
            C=c_val,
            solver="lbfgs",
            max_iter=self.epochs,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).flatten()

        self.model.fit(X, y)
        self.weights = self.model.coef_.T
        self.bias = self.model.intercept_.reshape(1, -1)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán phân phối xác suất của 10 chữ số"""
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Phân loại chữ số dựa trên lớp có xác suất cao nhất"""
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(X)