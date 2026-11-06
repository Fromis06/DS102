import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CATEGORY_LABELS = {"Nonfiction": -1, "Fiction": 1}


@dataclass(frozen=True)
class Book:
    title: str
    description: str
    category: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.description}"


def load_books(path: Path) -> list[Book]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, skipinitialspace=True)
        columns = [value.strip() for value in next(reader, [])]
        books = []
        for row in reader:
            if not row:
                continue
            fields = dict(zip(columns, (value.strip() for value in row)))
            if fields["category"] in CATEGORY_LABELS:
                books.append(Book(fields["title"], fields["description"], fields["category"]))
        return books


def split_by_category(books: list[Book], seed: int = 67) -> tuple[list[Book], list[Book]]:
    rng = np.random.default_rng(seed)
    train, test = [], []
    for category in CATEGORY_LABELS:
        indices = [index for index, book in enumerate(books) if book.category == category]
        shuffled = rng.permutation(indices)
        test_size = max(1, len(shuffled) // 5)
        train.extend(books[index] for index in shuffled[test_size:])
        test.extend(books[index] for index in shuffled[:test_size])
    return train, test
