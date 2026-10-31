import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, l2_penalty: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        m = len(y)
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1.0 - eps)
        
        bce_loss = - (1.0 / m) * np.sum(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
        l2_loss = (self.l2_penalty / (2.0 * m)) * np.sum(self.weights ** 2)
        return float(bce_loss + l2_loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        X: shape (m, n) với m là số mẫu, n là số feature
        y: shape (m,) hoặc (m, 1) với nhãn {0, 1}
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)
            dz = y_hat - y
            dw = (1.0 / m) * np.dot(X.T, dz) + (self.l2_penalty / m) * self.weights
            db = (1.0 / m) * np.sum(dz)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(y, y_hat)
            self.losses.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất mẫu thuộc class 1 (P(y=1|X))"""
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Phân loại nhị phân dựa trên ngưỡng threshold"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int).flatten()