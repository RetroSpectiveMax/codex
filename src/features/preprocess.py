"""Feature engineering utilities for the car reliability project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = 2024


@dataclass
class FeatureConfig:
    """Configuration describing modelling feature groups."""

    numeric_features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]
    text_feature: str


DEFAULT_FEATURE_CONFIG = FeatureConfig(
    numeric_features=(
        "mileage",
        "avg_trip_length_miles",
        "maintenance_events",
        "past_failures",
        "severity_score",
        "car_age",
        "total_cost_last_year",
    ),
    categorical_features=("make", "model", "maintenance_action"),
    text_feature="complaint_text",
)


COST_FACTORS = {
    "oil change": 120,
    "brake pad replacement": 380,
    "software update": 160,
    "battery inspection": 90,
    "tire rotation": 75,
}


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain specific columns used across the project."""
    engineered = df.copy()
    engineered["car_age"] = CURRENT_YEAR - engineered["year"]
    engineered["total_cost_last_year"] = (
        engineered["maintenance_cost_last_year"] + engineered["fuel_cost_last_year"]
    )
    issue_series = engineered.get("has_mechanical_issue", 0)
    engineered["estimated_next_year_cost"] = engineered["total_cost_last_year"] * (
        1.05 + 0.02 * issue_series
    )
    engineered["ownership_cost_score"] = (
        engineered["total_cost_last_year"] / engineered["avg_trip_length_miles"].clip(lower=1)
    )
    return engineered


def build_feature_transformer(config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> ColumnTransformer:
    """Create the column transformer for modelling."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    text_transformer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(config.numeric_features)),
            ("cat", categorical_transformer, list(config.categorical_features)),
            ("text", text_transformer, config.text_feature),
        ]
    )
    return preprocessor


def compute_cost_of_ownership(
    row: pd.Series,
    annual_mileage: int = 12000,
    fuel_price_per_gallon: float = 3.5,
    efficiency_mpg: float = 26,
) -> Dict[str, float]:
    """Estimate simplified cost-of-ownership metrics."""
    base_maintenance = COST_FACTORS.get(row.get("maintenance_action", ""), 150)
    risk_multiplier = 1.0 + 0.35 * row.get("risk_score", 0.2)
    maintenance_projection = (row.get("maintenance_cost_last_year", 600) + base_maintenance) * risk_multiplier

    fuel_cost_projection = (annual_mileage / efficiency_mpg) * fuel_price_per_gallon
    depreciation = max(500, 0.12 * row.get("total_cost_last_year", 1500))

    return {
        "maintenance_projection": round(maintenance_projection, 2),
        "fuel_cost_projection": round(fuel_cost_projection, 2),
        "depreciation_estimate": round(depreciation, 2),
        "total_projection": round(
            maintenance_projection + fuel_cost_projection + depreciation,
            2,
        ),
    }


def generate_maintenance_timeline(row: pd.Series) -> List[Dict[str, str]]:
    """Produce a simplified maintenance schedule based on risk."""
    base_interval = 6
    risk_score = row.get("risk_score", 0.25)
    compression = 1.0 - min(max(risk_score, 0.0), 0.95) * 0.4
    interval_months = max(3, int(base_interval * compression))

    milestones = [
        {
            "timeframe": f"{interval_months} months",
            "action": "Comprehensive inspection & fluid checks",
        },
        {
            "timeframe": f"{interval_months * 2} months",
            "action": "Predictive component diagnostics",
        },
        {
            "timeframe": f"{interval_months * 3} months",
            "action": "System software updates & alignment",
        },
    ]
    if risk_score > 0.6:
        milestones.append(
            {
                "timeframe": "Next 30 days",
                "action": "Schedule reliability assessment with specialist",
            }
        )
    return milestones
