"""Inference helpers for the reliability model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import pandas as pd

from src.features.preprocess import compute_cost_of_ownership, engineer_domain_features, generate_maintenance_timeline
from src.nlp.sentiment import append_sentiment_scores

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "reliability_model.joblib"


@dataclass
class PredictionResult:
    probability: float
    risk_band: str
    cost_projection: Dict[str, float]
    maintenance_timeline: List[Dict[str, str]]


class ReliabilityPredictor:
    """Load the persisted pipeline and expose helper utilities."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact missing at {path}. Train the model before predicting."
            )
        self.pipeline = joblib.load(path)

    @staticmethod
    def _prepare_frame(records: Iterable[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(list(records))
        df = engineer_domain_features(df)
        df = append_sentiment_scores(df)
        return df

    def predict(self, record: Dict) -> PredictionResult:
        frame = self._prepare_frame([record])
        proba = float(self.pipeline.predict_proba(frame)[0, 1])
        risk_band = self._to_risk_band(proba)
        frame["risk_score"] = proba
        cost_projection = compute_cost_of_ownership(frame.iloc[0])
        maintenance_timeline = generate_maintenance_timeline(frame.iloc[0])
        return PredictionResult(
            probability=proba,
            risk_band=risk_band,
            cost_projection=cost_projection,
            maintenance_timeline=maintenance_timeline,
        )

    def compare(self, record_a: Dict, record_b: Dict) -> Dict[str, PredictionResult]:
        return {
            "car_a": self.predict(record_a),
            "car_b": self.predict(record_b),
        }

    @staticmethod
    def _to_risk_band(probability: float) -> str:
        if probability >= 0.65:
            return "High"
        if probability >= 0.4:
            return "Moderate"
        return "Low"
