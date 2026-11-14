import matplotlib.pyplot as plt
import numpy as np

OUTPUT_LOSS_PATH = r"Thực hành\LAB3\BAI2\result\loss_comparison.png"
OUTPUT_PRED_PATH = r"Thực hành\LAB3\BAI2\result\standardize_comparison.png"

def plot_loss_comparison(losses_orig: list, losses_scaled: list, save_path: str = OUTPUT_LOSS_PATH):
    plt.figure(figsize=(10, 5))
    valid_orig = [l for l in losses_orig if np.isfinite(l)]
    if len(valid_orig) > 0:
        plt.plot(range(1, len(valid_orig) + 1), valid_orig, color="#d62728", linewidth=2, label="Original")
    plt.plot(range(1, len(losses_scaled) + 1), losses_scaled, color="#1f77b4", linewidth=2, label="Standardized")

    plt.title("So sánh Đường cong Hội tụ của Gốc vs Chuẩn hóa", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("loss_comparison.png", dpi=300)
    plt.close()

def plot_standardize_comparison(y_test: np.ndarray, pred_scaled: np.ndarray, save_path: str = OUTPUT_PRED_PATH):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred_scaled, color="#1f77b4", alpha=0.6, edgecolors="k", s=40, label="Dự đoán")

    min_val = min(float(np.min(y_test)), float(np.min(pred_scaled)))
    max_val = max(float(np.max(y_test)), float(np.max(pred_scaled)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = y_hat")

    plt.title("Giá trị Thực tế vs Dự đoán", fontsize=12, fontweight="bold")
    plt.xlabel("Thực tế", fontsize=11)
    plt.ylabel("Dự đoán", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("standardize_comparison.png", dpi=300)
    plt.close()
