import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_LOSS_PATH = r"Thực hành\LAB3\BAI3\result\model_comparison.png"
OUTPUT_CORR_PATH = r"Thực hành\LAB3\BAI3\result\correlation_matrix.png"
OUTPUT_PRED_PATH = r"Thực hành\LAB3\BAI3\result\prediction_comparison.png"
OUTPUT_WEIGHT_PATH = r"Thực hành\LAB3\BAI3\result\weights_comparison.png"

def plot_loss_comparison(losses_collinear: list, losses_clean: list, save_path: str = OUTPUT_LOSS_PATH):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(losses_collinear) + 1), losses_collinear, color="#d62728", linewidth=2, label="Dữ liệu Đa cộng tuyến (29 đặc trưng)")
    plt.plot(range(1, len(losses_clean) + 1), losses_clean, color="#1f77b4", linewidth=2, label="Sau Feature Engineering (26 đặc trưng)")
    plt.title("So sánh đường cong hội tụ loss của trước vs sau Feature Engineering", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("model_comparison.png", dpi=300)
    plt.close()

def plot_correlation_matrix(csv_path: str, save_path: str = OUTPUT_CORR_PATH):
    df = pd.read_csv(csv_path)
    num_cols = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.title("Ma trận tương quan đặc trưng số học", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("correlation_matrix.png", dpi=300)
    plt.close()

def plot_predictions(y_test: np.ndarray, pred_collinear: np.ndarray, pred_clean: np.ndarray, save_path: str = OUTPUT_PRED_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    min_val = min(float(np.min(y_test)), float(np.min(pred_collinear)), float(np.min(pred_clean)))
    max_val = max(float(np.max(y_test)), float(np.max(pred_collinear)), float(np.max(pred_clean)))

    # Đồ thị 1: Dữ liệu Đa cộng tuyến
    axes[0].scatter(y_test, pred_collinear, color="#d62728", alpha=0.6, edgecolors="k", s=40, label="Dự đoán (Đa cộng tuyến)")
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = y_hat")
    axes[0].set_title("Thực tế vs Dự đoán", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thực tế", fontsize=11)
    axes[0].set_ylabel("Dự đoán", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Đồ thị 2: Sau Feature Engineering
    axes[1].scatter(y_test, pred_clean, color="#1f77b4", alpha=0.6, edgecolors="k", s=40, label="Dự đoán (Sau Feature Engineering)")
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = y_hat")
    axes[1].set_title("Thực tế vs Dự đoán", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thực tế", fontsize=11)
    axes[1].set_ylabel("Dự đoán", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("prediction_comparison.png", dpi=300)
    plt.close()

def plot_weights_comparison(weights_collinear: np.ndarray, weights_clean: np.ndarray, feat_col: list, feat_clean: list, save_path: str = OUTPUT_WEIGHT_PATH):
    common_feats = [f for f in feat_clean if f in feat_col]
    w_col_dict = {feat_col[i]: float(weights_collinear[i, 0]) for i in range(len(feat_col))}
    w_cln_dict = {feat_clean[i]: float(weights_clean[i, 0]) for i in range(len(feat_clean))}
    w_col_vals = [w_col_dict[f] for f in common_feats]
    w_cln_vals = [w_cln_dict[f] for f in common_feats]
    y_pos = np.arange(len(common_feats))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y_pos - h/2, w_col_vals, h, label="Dữ liệu đa cộng tuyến", color="#d62728", alpha=0.75)
    ax.barh(y_pos + h/2, w_cln_vals, h, label="Sau Feature Engineering", color="#1f77b4", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(common_feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Giá trị trọng số")
    ax.set_title("So sánh phân bố trọng số của Trước vs Sau Feature Engineering", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
    except Exception:
        plt.savefig("weights_comparison.png", dpi=300)
    plt.close()
