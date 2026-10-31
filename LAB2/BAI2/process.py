import os
import gzip
import numpy as np

def _read_idx_images(gz_path: str) -> np.ndarray:
    """Đọc file ảnh (.gz) và trả về ma trận - Source: AI"""
    with gzip.open(gz_path, "rb") as f:
        # 16 byte đầu là header: magic_num (4), num_images (4), rows (4), cols (4)
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype(np.float64)

def _read_idx_labels(gz_path: str) -> np.ndarray:
    """Đọc file nhãn (.gz) và trả về mảng 1 chiều - Source: AI"""
    with gzip.open(gz_path, "rb") as f:
        # 8 byte đầu là header: magic_num (4), num_labels (4)
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels.astype(int)

def load_and_preprocess(data_dir: str):
    train_img_path = os.path.join(data_dir, "train_images.gz")
    train_lbl_path = os.path.join(data_dir, "train_labels.gz")
    test_img_path = os.path.join(data_dir, "test_images.gz")
    test_lbl_path = os.path.join(data_dir, "test_labels.gz")

    x_train = _read_idx_images(train_img_path)
    y_train = _read_idx_labels(train_lbl_path)
    x_test = _read_idx_images(test_img_path)
    y_test = _read_idx_labels(test_lbl_path)

    return x_train, x_test, y_train, y_test

def normalize_pixels(x_train: np.ndarray, x_test: np.ndarray):
    """Chuẩn hóa giá trị điểm ảnh"""
    return x_train / 255.0, x_test / 255.0

def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> dict:
    """Tính Precision, Recall, F1-score và Accuracy"""
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

    precisions = np.zeros(num_classes, dtype=np.float64)
    recalls = np.zeros(num_classes, dtype=np.float64)
    f1s = np.zeros(num_classes, dtype=np.float64)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions[c] = p
        recalls[c] = r
        f1s[c] = f1

        for pred_c in range(num_classes):
            confusion_matrix[c, pred_c] = int(np.sum((y_true == c) & (y_pred == pred_c)))

    accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "per_class_f1": f1s,
        "confusion_matrix": confusion_matrix,
    }