"""Model training pipeline for the reliability classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_reliability_data
from src.features.preprocess import (
    DEFAULT_FEATURE_CONFIG,
    build_feature_transformer,
    engineer_domain_features,
)
from src.nlp.sentiment import append_sentiment_scores

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
MODEL_PATH = ARTIFACT_DIR / "reliability_model.joblib"


class ReliabilityModelTrainer:
    """Handle training and evaluation of the gradient boosting model."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def load_dataset(self) -> pd.DataFrame:
        df = load_reliability_data()
        df = engineer_domain_features(df)
        df = append_sentiment_scores(df)
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        features = df.drop(columns=["has_mechanical_issue"])
        target = df["has_mechanical_issue"]
        return train_test_split(
            features,
            target,
            test_size=0.25,
            stratify=target,
            random_state=self.random_state,
        )

    def build_pipeline(self) -> Pipeline:
        preprocessor = build_feature_transformer(DEFAULT_FEATURE_CONFIG)
        model = GradientBoostingClassifier(random_state=self.random_state)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline = pipeline
        return pipeline

    def train(self) -> Dict[str, float]:
        df = self.load_dataset()
        X_train, X_test, y_train, y_test = self.split_data(df)

        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, predictions, output_dict=True)
        auc = roc_auc_score(y_test, probabilities)

        metrics = {
            "roc_auc": auc,
            "precision_high_risk": report["1"]["precision"],
            "recall_high_risk": report["1"]["recall"],
            "f1_high_risk": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
        }
        self.save_artifacts(pipeline, metrics)
        return metrics

    def save_artifacts(self, pipeline: Pipeline, metrics: Dict[str, float]) -> None:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        REPORTS_DIR.joinpath("training_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )


def main() -> None:
    trainer = ReliabilityModelTrainer()
    metrics = trainer.train()
    print("Training complete. Metrics saved to reports/training_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
