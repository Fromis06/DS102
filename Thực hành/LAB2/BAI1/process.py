import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path: str):
    """
    Đọc file books_to_scrape.csv và trích xuất ma trận đặc trưng x, nhãn nhị phân y.
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    
    # Nhãn 1: Sách có rating >= 3 sao
    # Nhãn 0: Sách có rating < 3 sao
    y = (df["rating"].values >= 3).astype(np.float64)

    # Trích xuất đặc trưng số
    price = df["price_excl_tax"].values.astype(np.float64)
    stock = df["stock_quantity"].values.astype(np.float64)
    title_len = df["title"].fillna("").str.len().values.astype(np.float64)
    desc_len = df["description"].fillna("").str.len().values.astype(np.float64)
    desc_words = (
        df["description"].fillna("").str.split().str.len().values.astype(np.float64)
    )

    x_num = np.column_stack([price, stock, title_len, desc_len, desc_words])

    # Phân loại category theo 0 và 1
    '''
        Giả sử trong cột category có 4 thể loại sách: Default, Fiction, Mystery, Poetry
        Nếu sách là Poetry thì cột Poetry = 1, các cột còn lại bằng 0.
        Nếu sách là Fiction thì cột Fiction = 1, các cột còn lại bằng 0.
        
        Kết quả:
        [0, 0, 0] đại diện cho Default
        [1, 0, 0] đại diện cho Fiction
        [0, 1, 0] đại diện cho Mystery
        [0, 0, 1] đại diện cho Poetry
    '''
    cat_dummies = pd.get_dummies(df["category"], drop_first=True).values.astype(np.float64)
    x = np.hstack([x_num, cat_dummies])
    
    return x, y

def train_test_split_numpy(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Chia tập dữ liệu"""
    np.random.seed(seed)
    m = len(y)
    indices = np.random.permutation(m)
    split_idx = int(m * (1.0 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def standardize_features(x_train: np.ndarray, x_test: np.ndarray):
    """Chuẩn hóa Z-score dựa trên tập Train"""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8
    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std
    return x_train_scaled, x_test_scaled

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính Precision, Recall, F1-score và Accuracy"""
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }