# Car Reliability Prediction Engine

This repository implements a prototype of the **Car Reliability Prediction Engine**, a system that ingests historical reliability signals, applies NLP to complaint text, and serves insights through an interactive dashboard.

## Project Highlights

- Gradient Boosting model trained on structured attributes and complaint text to predict mechanical issue risk by make/model/year.
- Complaint mining utilities that surface the most common failure patterns and risk-specific keywords.
- Lightweight sentiment analyser to gauge owner tone directly from complaint narratives.
- Simplified cost-of-ownership calculator and maintenance timeline forecaster.
- Streamlit dashboard for risk exploration and side-by-side vehicle comparison.

The current build uses a synthetic dataset that mimics the schema of NHTSA complaints and Consumer Reports ownership data. It is designed to be swapped with real-world datasets as they become available.

## Repository Structure

```
├── artifacts/                # Persisted model artifacts (created after training)
├── data/
│   └── car_reliability_synthetic.csv
├── docs/
├── reports/                  # Training metrics are written here
├── scripts/
│   └── generate_synthetic_data.py
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── predict.py
│   │   └── train.py
│   ├── nlp/
│   │   ├── complaint_analysis.py
│   │   └── sentiment.py
│   └── visualization/
│       └── dashboard.py
└── requirements.txt
```

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate or refresh synthetic data (optional)**

   ```bash
   python scripts/generate_synthetic_data.py
   ```

3. **Train the reliability model**

   ```bash
   python -m src.models.train
   ```

   A trained pipeline will be saved to `artifacts/reliability_model.joblib` with evaluation metrics in `reports/training_metrics.json`.

4. **Launch the dashboard**

   ```bash
   streamlit run src/visualization/dashboard.py
   ```

   Use the sidebar to explore different vehicles, review the predicted risk score, forecasted costs, maintenance timeline, and compare two cars directly.

## Data Source Notes

The synthetic dataset is shaped to integrate with the following real-world sources when available:

- [NHTSA Vehicle Complaint Data](https://www.nhtsa.gov/vehicle/complaints)
- [Consumer Reports Reliability Ratings](https://www.consumerreports.org/cars/)
- Owner forums such as `https://www.reddit.com/r/Cartalk/`, `https://www.carcomplaints.com/`, and manufacturer-specific communities (e.g., `https://www.teslamotorsclub.com/`).

Before shipping with production data, confirm access requirements, API rate limits, and licensing terms for each provider.

## Next Steps

- Replace the synthetic dataset with curated historical complaints and maintenance records.
- Experiment with model variants (e.g., XGBoost, LightGBM) and advanced embeddings for complaint narratives.
- Expose the predictor via a REST API for broader integration.
- Harden the dashboard for deployment and add mobile-responsive styling.
