import numpy as np
import pandas as pd

def load_and_preprocess_collinear_data(csv_path: str):
    """
    Tạo tập dữ liệu gốc chứa các cột phụ thuộc tuyến tính (Collinear):
    - Mã hóa One-hot đầy đủ không bỏ cột tham chiếu (drop_first=False), gây ra bẫy biến giả (Dummy Variable Trap):
      tổng các cột month = 1, tổng các cột day = 1, phụ thuộc tuyến tính vào cột bias.
    - Giữ lại các cột có tương quan rất cao (DC tương quan r = 0.682 với DMC).
    """
    df = pd.read_csv(csv_path)
    y = np.log1p(df["area"].values.astype(np.float64))

    df_features = df.drop(columns=["area"])
    df_collinear = pd.get_dummies(df_features, columns=["month", "day"], drop_first=False)

    x = df_collinear.values.astype(np.float64)
    feature_names = list(df_collinear.columns)
    return x, y, feature_names

def load_and_preprocess_clean_data(csv_path: str):
    """
    Thực hiện Feature Engineering để loại bỏ các cột phụ thuộc tuyến tính (Collinear):
    1. Tránh bẫy biến giả bằng drop_first=True.
    2. Loại bỏ cột DC có tương quan cao với DMC và các tháng mùa hè/thu.
    """
    df = pd.read_csv(csv_path)
    y = np.log1p(df["area"].values.astype(np.float64))

    df_features = df.drop(columns=["area"])
    df_clean = pd.get_dummies(df_features, columns=["month", "day"], drop_first=True)

    # Loại bỏ cột DC do đa cộng tuyến cao với DMC
    if "DC" in df_clean.columns:
        df_clean = df_clean.drop(columns=["DC"])

    x = df_clean.values.astype(np.float64)
    feature_names = list(df_clean.columns)
    return x, y, feature_names

def train_test_split_numpy(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Chia tập dữ liệu bằng NumPy"""
    np.random.seed(seed)
    m = len(y)
    indices = np.random.permutation(m)
    split_idx = int(m * (1.0 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def standardize_features(x_train: np.ndarray, x_test: np.ndarray):
    """Chuẩn hóa Z-score dựa trên tập Train"""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    x_train_scaled = (x_train - mean) / std_safe
    x_test_scaled = (x_test - mean) / std_safe
    return x_train_scaled, x_test_scaled

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính MSE, RMSE, MAE, R2-score"""
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
