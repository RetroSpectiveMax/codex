"""Streamlit dashboard for the Car Reliability Prediction Engine."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from src.features.preprocess import (
    compute_cost_of_ownership,
    engineer_domain_features,
    generate_maintenance_timeline,
)
from src.models.predict import MODEL_PATH, ReliabilityPredictor
from src.nlp.complaint_analysis import identify_common_failure_patterns, top_failure_terms_by_class
from src.nlp.sentiment import append_sentiment_scores

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "car_reliability_synthetic.csv"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = engineer_domain_features(df)
    df = append_sentiment_scores(df)
    return df


def render_sidebar_controls(df: pd.DataFrame) -> Dict:
    st.sidebar.header("Select vehicle")
    make = st.sidebar.selectbox("Make", sorted(df["make"].unique()))
    models = df.loc[df["make"] == make, "model"].unique()
    model = st.sidebar.selectbox("Model", sorted(models))
    years = df.loc[(df["make"] == make) & (df["model"] == model), "year"].unique()
    year = st.sidebar.selectbox("Model year", sorted(years, reverse=True))

    subset = df[(df["make"] == make) & (df["model"] == model) & (df["year"] == year)]
    baseline = subset.iloc[0]

    mileage = st.sidebar.slider("Mileage", 0, 200000, int(baseline["mileage"]))
    maintenance_events = st.sidebar.slider("Maintenance events", 0, 10, int(baseline["maintenance_events"]))
    past_failures = st.sidebar.slider("Past failures", 0, 6, int(baseline["past_failures"]))
    severity_score = st.sidebar.slider(
        "Severity score",
        0.0,
        10.0,
        float(baseline["severity_score"]),
    )

    complaint_text = st.sidebar.text_area("Complaint summary", baseline["complaint_text"], height=140)

    return {
        "make": make,
        "model": model,
        "year": year,
        "mileage": mileage,
        "avg_trip_length_miles": float(baseline["avg_trip_length_miles"]),
        "maintenance_events": maintenance_events,
        "past_failures": past_failures,
        "severity_score": severity_score,
        "maintenance_cost_last_year": float(baseline["maintenance_cost_last_year"]),
        "fuel_cost_last_year": float(baseline["fuel_cost_last_year"]),
        "maintenance_action": baseline["maintenance_action"],
        "complaint_text": complaint_text,
    }


def main() -> None:
    st.set_page_config(page_title="Car Reliability Prediction Engine", layout="wide")
    st.title("Car Reliability Prediction Engine")
    st.caption("Predict mechanical risk, understand failure patterns, and compare vehicles.")

    dataset = load_dataset()
    user_record = render_sidebar_controls(dataset)

    if MODEL_PATH.exists():
        predictor = ReliabilityPredictor()
        result = predictor.predict(user_record)
        st.subheader("Reliability Risk Score")
        st.metric("Probability of mechanical issue", f"{result.probability:.2%}", result.risk_band)

        st.subheader("Cost of Ownership Forecast")
        cost = result.cost_projection
        st.write(
            pd.DataFrame(
                {
                    "Metric": cost.keys(),
                    "USD": cost.values(),
                }
            )
        )

        st.subheader("Maintenance Timeline")
        for milestone in result.maintenance_timeline:
            st.write(f"**{milestone['timeframe']}** — {milestone['action']}")
    else:
        st.info("Train the model (python -m src.models.train) to enable live predictions.")

    st.divider()
    st.subheader("Common Failure Patterns")
    patterns = identify_common_failure_patterns(dataset)
    if patterns:
        st.dataframe(pd.DataFrame(patterns))
    else:
        st.write("Not enough complaint data to derive patterns.")

    st.subheader("Failure Terms by Risk Level")
    terms = top_failure_terms_by_class(dataset)
    for risk_level, tokens in terms.items():
        st.write(f"**{risk_level}:** {', '.join(tokens)}")

    st.subheader("Compare Two Vehicles")
    col1, col2 = st.columns(2)
    with col1:
        make_a = st.selectbox("Car A make", sorted(dataset["make"].unique()), key="a_make")
        model_a = st.selectbox(
            "Car A model",
            sorted(dataset.loc[dataset["make"] == make_a, "model"].unique()),
            key="a_model",
        )
        year_a = int(
            st.selectbox(
                "Car A year",
                sorted(dataset.loc[(dataset["make"] == make_a) & (dataset["model"] == model_a), "year"].unique()),
                key="a_year",
            )
        )
    with col2:
        make_b = st.selectbox("Car B make", sorted(dataset["make"].unique()), key="b_make")
        model_b = st.selectbox(
            "Car B model",
            sorted(dataset.loc[dataset["make"] == make_b, "model"].unique()),
            key="b_model",
        )
        year_b = int(
            st.selectbox(
                "Car B year",
                sorted(dataset.loc[(dataset["make"] == make_b) & (dataset["model"] == model_b), "year"].unique()),
                key="b_year",
            )
        )

    if st.button("Compare reliability", use_container_width=True):
        if MODEL_PATH.exists():
            predictor = ReliabilityPredictor()
            record_a = dataset[
                (dataset["make"] == make_a)
                & (dataset["model"] == model_a)
                & (dataset["year"] == year_a)
            ].iloc[0]
            record_b = dataset[
                (dataset["make"] == make_b)
                & (dataset["model"] == model_b)
                & (dataset["year"] == year_b)
            ].iloc[0]

            comparison = predictor.compare(record_a.to_dict(), record_b.to_dict())
            st.write(
                pd.DataFrame(
                    {
                        "Vehicle": ["Car A", "Car B"],
                        "Make": [make_a, make_b],
                        "Model": [model_a, model_b],
                        "Year": [year_a, year_b],
                        "Risk Probability": [
                            f"{comparison['car_a'].probability:.2%}",
                            f"{comparison['car_b'].probability:.2%}",
                        ],
                        "Risk Band": [
                            comparison["car_a"].risk_band,
                            comparison["car_b"].risk_band,
                        ],
                    }
                )
            )
        else:
            st.info("Train the model to enable comparison results.")


if __name__ == "__main__":
    main()
