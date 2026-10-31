import numpy as np
from SoftmaxRegression import SoftmaxRegression
import process

DATA_PATH = r"LAB2\data"

def main():
    X_train, X_test, y_train, y_test = process.load_and_preprocess(DATA_PATH)
    print(f"Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Kích thước X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"Các nhãn phân loại: {np.unique(y_train)}")

    X_train_scaled, X_test_scaled = process.normalize_pixels(X_train, X_test)

    model = SoftmaxRegression(
        learning_rate=0.5,
        epochs=300,
        l2_penalty=0.01,
        num_classes=10,
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_metrics = process.compute_multiclass_metrics(y_train, y_train_pred, num_classes=10)
    test_metrics = process.compute_multiclass_metrics(y_test, y_test_pred, num_classes=10)

    print(f"{'Metric (Macro)':<15} | {'Train Set':<20} | {'Test Set':<20}")
    print("-" * 55)
    print(f"{'Precision':<15} | {train_metrics['precision']:<20.4f} | {test_metrics['precision']:<20.4f}")
    print(f"{'Recall':<15} | {train_metrics['recall']:<20.4f} | {test_metrics['recall']:<20.4f}")
    print(f"{'F1-Score':<15} | {train_metrics['f1']:<20.4f} | {test_metrics['f1']:<20.4f}")
    print(f"{'Accuracy':<15} | {train_metrics['accuracy']:<20.4f} | {test_metrics['accuracy']:<20.4f}")

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])
    print()

if __name__ == "__main__":
    main()