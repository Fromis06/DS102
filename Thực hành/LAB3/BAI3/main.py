import os
import numpy as np
from LinearRegression import LinearRegression
import process
import visualize

DATA_PATH = r"Thực hành\LAB3\data\forestfires.csv" if os.path.exists(r"Thực hành\LAB3\data\forestfires.csv") else os.path.join("..", "data", "forestfires.csv")

def main():
    x_col, y_col, feat_col = process.load_and_preprocess_collinear_data(DATA_PATH)
    print(f"Kích thước X (Đa cộng tuyến): {x_col.shape}, Kích thước y: {y_col.shape}")
    x_cln, y_cln, feat_cln = process.load_and_preprocess_clean_data(DATA_PATH)
    print(f"Kích thước X (Sau Feature Engineering): {x_cln.shape}, Kích thước y: {y_cln.shape}\n")

    X_tr_col, X_te_col, y_tr_col, y_te_col = process.train_test_split_numpy(x_col, y_col, test_size=0.2, seed=42)
    X_tr_col_s, X_te_col_s = process.standardize_features(X_tr_col, X_te_col)

    X_tr_cln, X_te_cln, y_tr_cln, y_te_cln = process.train_test_split_numpy(x_cln, y_cln, test_size=0.2, seed=42)
    X_tr_cln_s, X_te_cln_s = process.standardize_features(X_tr_cln, X_te_cln)

    print("Huấn luyện Gradient Descent trên Dữ liệu Đa cộng tuyến:")
    model_col = LinearRegression(learning_rate=0.01, epochs=1000, l2_penalty=0.0)
    model_col.fit(X_tr_col_s, y_tr_col)
    print(f"Loss ở Epoch 1: {model_col.losses[0]:.6f}")
    print(f"Loss ở Epoch cuối: {model_col.epochs}: {model_col.losses[-1]:.6f}\n")

    print("Huấn luyện Gradient Descent trên Dữ liệu Sau Feature Engineering:")
    model_cln = LinearRegression(learning_rate=0.01, epochs=1000, l2_penalty=0.0)
    model_cln.fit(X_tr_cln_s, y_tr_cln)
    print(f"Loss ở Epoch 1: {model_cln.losses[0]:.6f}")
    print(f"Loss ở Epoch cuối: {model_cln.epochs}: {model_cln.losses[-1]:.6f}\n")

    pred_te_col = model_col.predict(X_te_col_s)
    pred_te_cln = model_cln.predict(X_te_cln_s)

    m_te_col = process.compute_regression_metrics(y_te_col, pred_te_col)
    m_te_cln = process.compute_regression_metrics(y_te_cln, pred_te_cln)

    norm_w_col = float(np.linalg.norm(model_col.weights))
    norm_w_cln = float(np.linalg.norm(model_cln.weights))

    print(f"{'Metric':<20} | {'Dữ liệu Đa cộng tuyến':<25} | {'Sau Feature Engineering':<25}")
    print("-" * 75)
    print(f"{'Số đặc trưng':<20} | {x_col.shape[1]:<25} | {x_cln.shape[1]:<25}")
    print(f"{'Độ lớn trọng số':<20} | {norm_w_col:<25.4f} | {norm_w_cln:<25.4f}")
    print(f"{'Test MSE':<20} | {m_te_col['mse']:<25.4f} | {m_te_cln['mse']:<25.4f}")
    print(f"{'Test RMSE':<20} | {m_te_col['rmse']:<25.4f} | {m_te_cln['rmse']:<25.4f}")
    print(f"{'Test MAE':<20} | {m_te_col['mae']:<25.4f} | {m_te_cln['mae']:<25.4f}")
    print(f"{'Test R2-Score':<20} | {m_te_col['r2']:<25.4f} | {m_te_cln['r2']:<25.4f}")
    visualize.plot_loss_comparison(model_col.losses, model_cln.losses)
    visualize.plot_correlation_matrix(DATA_PATH)
    visualize.plot_predictions(y_te_col, pred_te_col, pred_te_cln)
    visualize.plot_weights_comparison(model_col.weights, model_cln.weights, feat_col, feat_cln)

if __name__ == "__main__":
    main()
