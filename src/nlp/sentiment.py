"""Lightweight sentiment scoring for owner feedback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

POSITIVE_WORDS = {
    "reliable",
    "smooth",
    "satisfied",
    "comfortable",
    "quiet",
    "refined",
    "responsive",
}

NEGATIVE_WORDS = {
    "frustrating",
    "disappointed",
    "unsafe",
    "annoying",
    "expensive",
    "noisy",
    "rough",
    "lag",
    "stall",
    "failure",
}


@dataclass
class SentimentScores:
    positive: int
    negative: int
    net: int


def score_text(text: str) -> SentimentScores:
    words = {word.strip(".,!?").lower() for word in text.split()}
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return SentimentScores(positive=pos, negative=neg, net=pos - neg)


def append_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched[["sentiment_positive", "sentiment_negative", "sentiment_net"]] = df[
        "complaint_text"
    ].apply(lambda text: pd.Series(score_text(text).__dict__))
    return enriched
