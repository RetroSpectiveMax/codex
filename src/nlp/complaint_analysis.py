"""NLP helpers for extracting failure patterns from complaint text."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _tokenize_phrases(corpus: Iterable[str], ngram_range: Tuple[int, int]) -> Counter:
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(corpus)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    return Counter(dict(zip(terms, counts)))


def identify_common_failure_patterns(
    df: pd.DataFrame, top_n: int = 8
) -> List[Dict[str, float]]:
    """Return frequently occurring complaint phrases weighted by severity."""
    if df.empty:
        return []

    counter = _tokenize_phrases(df["complaint_text"], ngram_range=(2, 3))
    severity = df["severity_score"].to_numpy()

    weighted_counts = Counter()
    for phrase, count in counter.items():
        weighted_counts[phrase] = count * (severity.mean() / max(severity.std(), 0.5))

    most_common = weighted_counts.most_common(top_n)
    return [
        {"pattern": phrase, "weighted_score": round(score, 2)}
        for phrase, score in most_common
    ]


def top_failure_terms_by_class(
    df: pd.DataFrame, target_col: str = "has_mechanical_issue", top_n: int = 5
) -> Dict[str, List[str]]:
    """Identify discriminative terms for each reliability class."""
    if df.empty:
        return {}

    vectorizer = CountVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(df["complaint_text"])
    terms = np.array(vectorizer.get_feature_names_out())

    results: Dict[str, List[str]] = {}
    for target_value in sorted(df[target_col].unique()):
        mask = df[target_col] == target_value
        counts = np.asarray(matrix[mask].sum(axis=0)).ravel()
        top_indices = counts.argsort()[::-1][:top_n]
        label = "High Risk" if target_value == 1 else "Low Risk"
        results[label] = terms[top_indices].tolist()
    return results
