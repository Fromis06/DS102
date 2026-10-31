import numpy as np
from LogisticRegression import LogisticRegression
import process
import visualize

DATA_PATH = r"LAB2\data\books_to_scrape.csv"

def main():
    x, y = process.load_and_preprocess_data(DATA_PATH)
    print(f"Kích thước X: {x.shape}, Kích thước y: {y.shape}")
    print(f"Phân bố nhãn y: Class 1 = {int(np.sum(y == 1))}, Class 0 = {int(np.sum(y == 0))}")

    X_train, X_test, y_train, y_test = process.train_test_split_numpy(x, y, test_size=0.2, seed=36)
    X_train_scaled, X_test_scaled = process.standardize_features(X_train, X_test)

    model = LogisticRegression(learning_rate=0.05, epochs=1000, l2_penalty=0.1)
    model.fit(X_train_scaled, y_train)
    print(f"Loss ở Epoch 1: {model.losses[0]:.6f}")
    print(f"Loss ở Epoch cuối: {model.epochs}: {model.losses[-1]:.6f}\n")

    y_train_pred = model.predict(X_train_scaled, threshold=0.5)
    y_test_pred = model.predict(X_test_scaled, threshold=0.5)

    train_metrics = process.compute_classification_metrics(y_train, y_train_pred)
    test_metrics = process.compute_classification_metrics(y_test, y_test_pred)

    print(f"{'Metric':<15} | {'Train Set':<20} | {'Test Set':<20}")
    print("-" * 50)
    print(f"{'Precision':<15} | {train_metrics['precision']:<20.4f} | {test_metrics['precision']:<20.4f}")
    print(f"{'Recall':<15} | {train_metrics['recall']:<20.4f} | {test_metrics['recall']:<20.4f}")
    print(f"{'F1-Score':<15} | {train_metrics['f1']:<20.4f} | {test_metrics['f1']:<20.4f}")
    print(f"{'Accuracy':<15} | {train_metrics['accuracy']:<20.4f} | {test_metrics['accuracy']:<20.4f}")
    print(
        f"\nConfusion Matrix (Test): TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
        f"TN={test_metrics['tn']}, FN={test_metrics['fn']}\n"
    )
    visualize.plot_loss_curve(model.losses)

if __name__ == "__main__":
    main()