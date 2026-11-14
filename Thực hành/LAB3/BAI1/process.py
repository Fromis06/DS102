import numpy as np
import pandas as pd

def load_and_preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)
    y = np.log1p(df["area"].values.astype(np.float64))
    df_features = df.drop(columns=["area"])
    df_encoded = pd.get_dummies(df_features, columns=["month", "day"], drop_first=True)
    x = df_encoded.values.astype(np.float64)
    return x, y

def train_test_split_numpy(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    np.random.seed(seed)
    m = len(y)
    indices = np.random.permutation(m)
    split_idx = int(m * (1.0 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def standardize_features(x_train: np.ndarray, x_test: np.ndarray):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8
    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std
    return x_train_scaled, x_test_scaled

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    n = len(y_true)
    if n == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    errors = y_true - y_pred
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
