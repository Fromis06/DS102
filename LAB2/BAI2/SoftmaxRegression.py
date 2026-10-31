import numpy as np

class SoftmaxRegression:
    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 300,
        l2_penalty: float = 0.0,
        num_classes: int = 10,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.num_classes = num_classes
        self.weights = None
        self.bias = None
        self.losses = []

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Tính xác suất Softmax"""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """Chuyển nhãn rời rạc"""
        y_int = np.asarray(y, dtype=int).flatten()
        return np.eye(self.num_classes, dtype=np.float64)[y_int]

    def _compute_loss(self, y_onehot: np.ndarray, y_hat: np.ndarray) -> float:
        m = len(y_onehot)
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1.0 - eps)

        cce_loss = -(1.0 / m) * np.sum(y_onehot * np.log(y_hat))
        l2_loss = (self.l2_penalty / (2.0 * m)) * np.sum(self.weights ** 2)
        return float(cce_loss + l2_loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=np.float64)
        y_onehot = self._one_hot(y)

        m, n = X.shape
        self.weights = np.zeros((n, self.num_classes), dtype=np.float64)
        self.bias = np.zeros((1, self.num_classes), dtype=np.float64)
        self.losses = []

        for _ in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._softmax(z)
            dz = y_hat - y_onehot
            dw = (1.0 / m) * np.dot(X.T, dz) + (self.l2_penalty / m) * self.weights
            db = (1.0 / m) * np.sum(dz, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(y_onehot, y_hat)
            self.losses.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán phân phối xác suất của 10 chữ số"""
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.weights) + self.bias
        return self._softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Phân loại chữ số dựa trên lớp có xác suất cao nhất"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    