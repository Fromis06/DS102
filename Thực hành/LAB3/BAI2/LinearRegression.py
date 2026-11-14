import numpy as np

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, l2_penalty: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None
        self.losses = []

    def _compute_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        m = len(y)
        mse_loss = (1.0 / (2.0 * m)) * np.sum((y - y_hat) ** 2)
        l2_loss = (self.l2_penalty / (2.0 * m)) * np.sum(self.weights ** 2) if self.weights is not None else 0.0
        loss = float(mse_loss + l2_loss)
        return loss if np.isfinite(loss) else float("inf")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        m, n = X.shape
        self.weights = np.zeros((n, 1), dtype=np.float64)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            y_hat = np.dot(X, self.weights) + self.bias
            error = y_hat - y 

            dw = (1.0 / m) * np.dot(X.T, error) + (self.l2_penalty / m) * self.weights
            db = (1.0 / m) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(y, y_hat)
            self.losses.append(loss)
            if not np.isfinite(loss):
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y_hat = np.dot(X, self.weights) + self.bias
        return y_hat.flatten()
