import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PRED_PATH = r"Thực hành\LAB3\BAI4\result\manual_vs_sklearn.png"
OUTPUT_WEIGHT_PATH = r"Thực hành\LAB3\BAI4\result\weights_comparison.png"

def plot_manual_vs_sklearn(y_test: np.ndarray, pred_manual: np.ndarray, pred_sklearn: np.ndarray, save_path: str = OUTPUT_PRED_PATH):
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_manual, pred_sklearn, color="#9467bd", alpha=0.7, edgecolors="k", s=40, label="Dự đoán mẫu test")

    min_v = float(min(np.min(pred_manual), np.min(pred_sklearn)))
    max_v = float(max(np.max(pred_manual), np.max(pred_sklearn)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Đường khớp y = x")

    plt.title("Đối chiếu Dự đoán: Manual NumPy (GD) vs Scikit-Learn (SGDRegressor)", fontsize=12, fontweight="bold")
    plt.xlabel("Dự đoán Manual NumPy", fontsize=11)
    plt.ylabel("Dự đoán Scikit-Learn", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("manual_vs_sklearn.png", dpi=300)
    plt.close()

def plot_weights_comparison(weights_manual: np.ndarray, weights_sklearn: np.ndarray, feature_names: list, save_path: str = OUTPUT_WEIGHT_PATH):
    y_pos = np.arange(len(feature_names))
    h = 0.4

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y_pos - h/2, weights_manual.flatten(), h, label="NumPy thuần", color="#1f77b4", alpha=0.85)
    ax.barh(y_pos + h/2, weights_sklearn.flatten(), h, label="Scikit-Learn", color="#ff7f0e", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Giá trị trọng số")
    ax.set_title("So sánh trọng số của NumPy thuần vs Scikit-Learn", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("weights_comparison.png", dpi=300)
    plt.close()
