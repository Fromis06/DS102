import matplotlib.pyplot as plt

OUTPUT_PATH = r"LAB2\BAI1\result\graph.png"

def plot_loss_curve(losses: list, save_path: str = OUTPUT_PATH):
    """Trực quan hóa quá trình giảm hàm Loss qua từng epoch"""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color="#1f77b4", linewidth=2, label="BCE + L2 Loss")
    plt.title("Training Loss Curve - Logistic Regression", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()