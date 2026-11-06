import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

WORD = re.compile(r"\b\w\w+\b")

class TfidfVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.idf = None

    def fit(self, documents: Sequence[str]):
        document_frequency = Counter()
        for text in documents:
            document_frequency.update(set(WORD.findall(text.lower())))
        words = sorted(document_frequency)
        self.vocabulary = {word: index for index, word in enumerate(words)}
        frequencies = np.array([document_frequency[word] for word in words])
        self.idf = 1 + np.log((1 + len(documents)) / (1 + frequencies))
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        features = np.zeros((len(documents), len(self.vocabulary)))
        for index, text in enumerate(documents):
            for word, count in Counter(WORD.findall(text.lower())).items():
                if word in self.vocabulary:
                    features[index, self.vocabulary[word]] = count

        features *= self.idf
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.where(norms > 0, norms, 1)
