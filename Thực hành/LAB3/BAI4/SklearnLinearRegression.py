import numpy as np
from sklearn.linear_model import SGDRegressor

class SklearnLinearRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, l2_penalty: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None

        penalty = "l2" if l2_penalty > 0 else None
        alpha = l2_penalty if l2_penalty > 0 else 0.0001
        self.model = SGDRegressor(
            loss="squared_error",
            penalty=penalty,
            alpha=alpha,
            learning_rate="invscaling",
            eta0=learning_rate,
            max_iter=epochs,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLinearRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        self.model.fit(X, y)
        self.weights = self.model.coef_.reshape(-1, 1)
        self.bias = float(self.model.intercept_[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(X).flatten()
