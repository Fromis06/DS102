import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, l2_penalty: float = 0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None

        c_val = 1.0 / self.l2_penalty if self.l2_penalty > 0 else 1e10

        self.model = SklearnLogisticRegression(
            penalty="l2",
            C=c_val,
            solver="lbfgs",
            max_iter=self.epochs,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).flatten()

        self.model.fit(X, y)
        self.weights = self.model.coef_.T
        self.bias = float(self.model.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int).flatten()