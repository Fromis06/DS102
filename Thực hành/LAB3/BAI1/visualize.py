import matplotlib.pyplot as plt
import numpy as np

OUTPUT_LOSS_PATH = r"Thực hành\LAB3\BAI1\result\loss_curve.png"
OUTPUT_PRED_PATH = r"Thực hành\LAB3\BAI1\result\prediction_eval.png"

def plot_loss_curve(losses: list, save_path: str = OUTPUT_LOSS_PATH):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color="#1f77b4", linewidth=2, label="MSE Loss")
    plt.title("Training Loss Curve - Linear Regression (Gradient Descent)", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("loss_curve.png", dpi=300)
    plt.close()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = OUTPUT_PRED_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, color="#2ca02c", edgecolors="k", s=40, label="Dự đoán")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="y = y_hat")
    axes[0].set_title("Giá trị Thực tế vs Dự đoán", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thực tế", fontsize=11)
    axes[0].set_ylabel("Dự đoán", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=25, color="#ff7f0e", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Phân phối Phần dư", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Phần dư", fontsize=11)
    axes[1].set_ylabel("Tần suất", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("prediction_eval.png", dpi=300)
    plt.close()
