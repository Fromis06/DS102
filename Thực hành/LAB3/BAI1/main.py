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

    model = LinearRegression(learning_rate=0.01, epochs=1000, l2_penalty=0.0)
    model.fit(X_train_scaled, y_train)
    print(f"Loss ở Epoch 1: {model.losses[0]:.6f}")
    print(f"Loss ở Epoch cuối: {model.epochs}: {model.losses[-1]:.6f}\n")

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_metrics = process.compute_regression_metrics(y_train, y_train_pred)
    test_metrics = process.compute_regression_metrics(y_test, y_test_pred)

    print(f"{'Metric':<15} | {'Train Set':<20} | {'Test Set':<20}")
    print("-" * 50)
    print(f"{'MSE':<15} | {train_metrics['mse']:<20.4f} | {test_metrics['mse']:<20.4f}")
    print(f"{'RMSE':<15} | {train_metrics['rmse']:<20.4f} | {test_metrics['rmse']:<20.4f}")
    print(f"{'MAE':<15} | {train_metrics['mae']:<20.4f} | {test_metrics['mae']:<20.4f}")
    print(f"{'R2-Score':<15} | {train_metrics['r2']:<20.4f} | {test_metrics['r2']:<20.4f}")

    visualize.plot_loss_curve(model.losses)
    visualize.plot_predictions(y_test, y_test_pred)

if __name__ == "__main__":
    main()
