import os
import numpy as np
from LinearRegression import LinearRegression
import process
import visualize

DATA_PATH = r"Thực hành\LAB3\data\forestfires.csv"

def main():
    x, y = process.load_and_preprocess_data(DATA_PATH)
    print(f"Kích thước X: {x.shape}, Kích thước y: {y.shape}")

    X_train, X_test, y_train, y_test = process.train_test_split_numpy(x, y, test_size=0.2, seed=42)
    X_train_scaled, X_test_scaled = process.standardize_features(X_train, X_test)

    print(f"Mean trung bình:     {np.mean(X_train_scaled):.6f}")
    print(f"Variance trung bình: {np.mean(np.var(X_train_scaled, axis=0)):.6f}\n")

    model_orig = LinearRegression(learning_rate=0.01, epochs=300, l2_penalty=0.0)
    model_orig.fit(X_train, y_train)
    print(f"Loss ở Epoch 1: {model_orig.losses[0]:.6f}")
    loss_orig_end = model_orig.losses[-1] if len(model_orig.losses) > 0 else float("nan")
    print(f"Loss ở Epoch cuối (300): {loss_orig_end}")
    model_scaled = LinearRegression(learning_rate=0.01, epochs=300, l2_penalty=0.0)
    model_scaled.fit(X_train_scaled, y_train)
    print(f"Loss ở Epoch 1: {model_scaled.losses[0]:.6f}")
    print(f"Loss ở Epoch cuối (300): {model_scaled.losses[-1]:.6f}\n")

    y_train_pred_scaled = model_scaled.predict(X_train_scaled)
    y_test_pred_scaled = model_scaled.predict(X_test_scaled)

    train_metrics = process.compute_regression_metrics(y_train, y_train_pred_scaled)
    test_metrics = process.compute_regression_metrics(y_test, y_test_pred_scaled)

    print(f"{'Metric':<15} | {'Dữ liệu Chuẩn hóa (Train)':<28} | {'Dữ liệu Chuẩn hóa (Test)':<28}")
    print("-" * 75)
    print(f"{'MSE':<15} | {train_metrics['mse']:<28.4f} | {test_metrics['mse']:<28.4f}")
    print(f"{'RMSE':<15} | {train_metrics['rmse']:<28.4f} | {test_metrics['rmse']:<28.4f}")
    print(f"{'MAE':<15} | {train_metrics['mae']:<28.4f} | {test_metrics['mae']:<28.4f}")
    print(f"{'R2-Score':<15} | {train_metrics['r2']:<28.4f} | {test_metrics['r2']:<28.4f}")

    visualize.plot_loss_comparison(model_orig.losses, model_scaled.losses)
    visualize.plot_standardize_comparison(y_test, y_test_pred_scaled)

if __name__ == "__main__":
    main()
