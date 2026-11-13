"""Model loading, feature encoding and the business rules around a churn score."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Decision threshold tuned in notebooks/01_EDA_and_modeling.ipynb.
# Below 0.50 on purpose: missing a churner costs more than an extra retention offer.
THRESHOLD = 0.40

MODEL_FACTS = {
    "Algorithm": "Logistic Regression",
    "ROC-AUC (test)": "0.840",
    "Decision threshold": "0.40",
    "Training customers": "5,634",
}

_TRI = {"No internet service": 0, "No": 1, "Yes": 2}


@dataclass(frozen=True)
class Artifacts:
    model: object
    scaler: object
    feature_names: list[str]


def load_artifacts() -> Artifacts:
    return Artifacts(
        model=joblib.load(MODEL_DIR / "churn_model.pkl"),
        scaler=joblib.load(MODEL_DIR / "scaler.pkl"),
        feature_names=list(joblib.load(MODEL_DIR / "feature_names.pkl")),
    )


def encode(c: dict) -> dict:
    """Map form answers to the label-encoded features the model was trained on."""
    return {
        "gender": 1 if c["gender"] == "Male" else 0,
        "SeniorCitizen": 1 if c["senior"] == "Yes" else 0,
        "Partner": 1 if c["partner"] == "Yes" else 0,
        "Dependents": 1 if c["dependents"] == "Yes" else 0,
        "tenure": c["tenure"],
        "PhoneService": 1 if c["phone_service"] == "Yes" else 0,
        "MultipleLines": {"No phone service": 0, "No": 1, "Yes": 2}[c["multiple_lines"]],
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[c["internet_service"]],
        "OnlineSecurity": _TRI[c["online_security"]],
        "OnlineBackup": _TRI[c["online_backup"]],
        "DeviceProtection": _TRI[c["device_protection"]],
        "TechSupport": _TRI[c["tech_support"]],
        "StreamingTV": _TRI[c["streaming_tv"]],
        "StreamingMovies": _TRI[c["streaming_movies"]],
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[c["contract"]],
        "PaperlessBilling": 1 if c["paperless_billing"] == "Yes" else 0,
        "PaymentMethod": {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3,
        }[c["payment_method"]],
        "MonthlyCharges": c["monthly_charges"],
        # Total can't be lower than one month's bill.
        "TotalCharges": max(c["total_charges"], c["monthly_charges"]),
    }


def predict(art: Artifacts, features: dict) -> float:
    row = pd.DataFrame([features])[art.feature_names].fillna(0)
    return float(art.model.predict_proba(art.scaler.transform(row))[0][1])


def risk_tier(prob: float) -> tuple[str, str]:
    """Return (label, streamlit color) for a churn probability."""
    if prob < 0.30:
        return "Low risk", "green"
    if prob < 0.60:
        return "Medium risk", "orange"
    return "High risk", "red"


def risk_factors(c: dict) -> list[tuple[str, str]]:
    """Rule-based explanations drawn from the EDA findings."""
    f = []
    if c["contract"] == "Month-to-month":
        f.append(("High", "Month-to-month contract: 42.7% churned in the data, vs 2.8% on two-year plans."))
    if c["tenure"] < 12:
        f.append(("High", f"Tenure of {c['tenure']} months: the median churner left at 10 months."))
    if c["tech_support"] == "No":
        f.append(("Medium", "No tech support: one of the strongest service-level churn signals."))
    if c["online_security"] == "No":
        f.append(("Medium", "No online security: customers without it churned far more often."))
    if c["internet_service"] == "Fiber optic":
        f.append(("Medium", "Fiber optic: a historically high-churn segment in this dataset."))
    if c["monthly_charges"] > 75:
        f.append(("Medium", f"Monthly bill of ${c['monthly_charges']:.0f}: churners averaged $74.44."))
    if c["paperless_billing"] == "Yes":
        f.append(("Low", "Paperless billing: a mild positive link with churn."))
    return f


def retention_actions(c: dict, prob: float) -> list[str]:
    if prob < THRESHOLD:
        return []
    a = []
    if c["contract"] == "Month-to-month":
        a.append("Offer a loyalty discount for moving to a one-year contract.")
    if c["tech_support"] == "No":
        a.append("Offer a free 30-day tech support trial.")
    if c["online_security"] == "No":
        a.append("Bundle online security at a reduced rate.")
    if c["tenure"] < 12:
        a.append("Assign an onboarding contact for the first year.")
    if c["monthly_charges"] > 75:
        a.append("Review the plan: a cheaper tier may fit this customer better.")
    return a[:4]


def what_if(art: Artifacts, features: dict) -> list[tuple[str, float]]:
    """Churn probability under each single retention change."""
    scenarios = {
        "One-year contract": dict(features, Contract=1),
        "Two-year contract": dict(features, Contract=2),
        "Add tech support": dict(features, TechSupport=2),
        "Add online security": dict(features, OnlineSecurity=2),
    }
    return [(name, predict(art, s)) for name, s in scenarios.items()]
