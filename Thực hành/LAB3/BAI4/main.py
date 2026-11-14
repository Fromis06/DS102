import os
import time
import numpy as np
from LinearRegression import LinearRegression as ManualLinearRegression
from SklearnLinearRegression import SklearnLinearRegression
import process
import visualize

DATA_PATH = r"Thực hành\LAB3\data\forestfires.csv" if os.path.exists(r"Thực hành\LAB3\data\forestfires.csv") else os.path.join("..", "data", "forestfires.csv")

def main():
    x, y, feature_names = process.load_and_preprocess_data(DATA_PATH)
    print(f"Kích thước X: {x.shape}, Kích thước y: {y.shape}")

    X_train, X_test, y_train, y_test = process.train_test_split_numpy(x, y, test_size=0.2, seed=42)
    X_train_scaled, X_test_scaled = process.standardize_features(X_train, X_test)
    t0 = time.perf_counter()
    model_manual = ManualLinearRegression(learning_rate=0.01, epochs=1000, l2_penalty=0.0)
    model_manual.fit(X_train_scaled, y_train)
    time_manual = (time.perf_counter() - t0) * 1000
    print(f"Loss Manual (GD) ở Epoch 1:    {model_manual.losses[0]:.6f}")
    print(f"Loss Manual (GD) ở Epoch cuối: {model_manual.losses[-1]:.6f}")
    t0 = time.perf_counter()
    model_sklearn = SklearnLinearRegression(learning_rate=0.01, epochs=1000, l2_penalty=0.0)
    model_sklearn.fit(X_train_scaled, y_train)
    time_sklearn = (time.perf_counter() - t0) * 1000
    y_train_pred_man = model_manual.predict(X_train_scaled)
    y_test_pred_man = model_manual.predict(X_test_scaled)

    y_train_pred_skl = model_sklearn.predict(X_train_scaled)
    y_test_pred_skl = model_sklearn.predict(X_test_scaled)

    m_tr_man = process.compute_regression_metrics(y_train, y_train_pred_man)
    m_te_man = process.compute_regression_metrics(y_test, y_test_pred_man)

    m_tr_skl = process.compute_regression_metrics(y_train, y_train_pred_skl)
    m_te_skl = process.compute_regression_metrics(y_test, y_test_pred_skl)
    print(f"{'Metric':<25} | {'Manual NumPy (GD)':<22} | {'Scikit-Learn (SGD)':<22}")
    print("-" * 75)
    print(f"{'Thời gian train (ms)':<25} | {time_manual:<22.3f} | {time_sklearn:<22.3f}")
    print(f"{'Bias / Intercept':<25} | {model_manual.bias:<22.4f} | {model_sklearn.bias:<22.4f}")
    print(f"{'Train MSE':<25} | {m_tr_man['mse']:<22.4f} | {m_tr_skl['mse']:<22.4f}")
    print(f"{'Train RMSE':<25} | {m_tr_man['rmse']:<22.4f} | {m_tr_skl['rmse']:<22.4f}")
    print(f"{'Train MAE':<25} | {m_tr_man['mae']:<22.4f} | {m_tr_skl['mae']:<22.4f}")
    print(f"{'Train R2-Score':<25} | {m_tr_man['r2']:<22.4f} | {m_tr_skl['r2']:<22.4f}")
    print("-" * 75)
    print(f"{'Test MSE':<25} | {m_te_man['mse']:<22.4f} | {m_te_skl['mse']:<22.4f}")
    print(f"{'Test RMSE':<25} | {m_te_man['rmse']:<22.4f} | {m_te_skl['rmse']:<22.4f}")
    print(f"{'Test MAE':<25} | {m_te_man['mae']:<22.4f} | {m_te_skl['mae']:<22.4f}")
    print(f"{'Test R2-Score':<25} | {m_te_man['r2']:<22.4f} | {m_te_skl['r2']:<22.4f}")
    visualize.plot_manual_vs_sklearn(y_test, y_test_pred_man, y_test_pred_skl)
    visualize.plot_weights_comparison(model_manual.weights, model_sklearn.weights, feature_names)

if __name__ == "__main__":
    main()
