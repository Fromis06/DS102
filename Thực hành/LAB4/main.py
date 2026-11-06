import argparse
from pathlib import Path

import numpy as np

from book_data import CATEGORY_LABELS, load_books, split_by_category
from smo import LinearSVM
from tfidf import TfidfVectorizer


DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "books_to_scrape.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Phân loại sách Fiction và Nonfiction bằng TF-IDF và SMO."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Đường dẫn đến tệp CSV")
    parser.add_argument("--c", type=float, default=2.0, help="Hệ số phạt vi phạm biên")
    parser.add_argument("--seed", type=int, default=42, help="Hãy chọn số đúng")
    parser.add_argument("--title", default="", help="Tiêu đề sách cần phân loại")
    parser.add_argument("--description", default="", help="Mô tả sách cần phân loại")
    args = parser.parse_args()

    books = load_books(args.csv)
    train, test = split_by_category(books, seed=args.seed)
    train_text = [book.text for book in train]
    vectorizer = TfidfVectorizer().fit(train_text)
    train_features = vectorizer.transform(train_text)
    test_features = vectorizer.transform([book.text for book in test])
    train_labels = np.array([CATEGORY_LABELS[book.category] for book in train])
    test_labels = np.array([CATEGORY_LABELS[book.category] for book in test])
    model = LinearSVM(c=args.c).fit(train_features, train_labels)

    predicted = model.predict(test_features)
    correct = np.count_nonzero(predicted == test_labels)
    majority_label = max(
        CATEGORY_LABELS.values(), key=lambda label: np.count_nonzero(train_labels == label)
    )
    baseline = np.count_nonzero(test_labels == majority_label)

    print(f"Huấn luyện: {len(train)} | Kiểm tra: {len(test)} | Số lần SMO cập nhật cặp: {model.updates}")
    print(f"Độ chính xác: {correct}/{len(test)} ({correct / len(test):.1%})")
    print(f"Luôn đoán lớp đông nhất: {baseline}/{len(test)} ({baseline / len(test):.1%})")
    print(f"{'Nhãn thật':<14} {'Đoán Nonfiction':>20} {'Đoán Fiction':>18}")
    for category, label in CATEGORY_LABELS.items():
        counts = [
            np.count_nonzero((test_labels == label) & (predicted == value))
            for value in CATEGORY_LABELS.values()
        ]
        print(f"{category:<14} {counts[0]:>20} {counts[1]:>18}")

    if args.title or args.description:
        features = vectorizer.transform([f"{args.title} {args.description}"])
        category_names = {label: name for name, label in CATEGORY_LABELS.items()}
        print(f"Danh mục dự đoán: {category_names[model.predict(features)[0]]}")

if __name__ == "__main__":
    main()
