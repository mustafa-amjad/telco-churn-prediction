import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight — Retention Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
model        = joblib.load(BASE_DIR / "model" / "churn_model.pkl")
scaler       = joblib.load(BASE_DIR / "model" / "scaler.pkl")
feature_names = joblib.load(BASE_DIR / "model" / "feature_names.pkl")
THRESHOLD    = 0.40  # tuned threshold from notebook

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, .stApp {
    background-color: #080d14;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1520;
    border-right: 1px solid rgba(99,179,237,0.12);
}

/* ── Hero header ── */
.hero-wrap {
    padding: 40px 0 28px 0;
    text-align: center;
    border-bottom: 1px solid rgba(99,179,237,0.10);
    margin-bottom: 32px;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f0f6ff;
    line-height: 1.15;
    margin: 0;
}
.hero-title span {
    background: linear-gradient(90deg, #63b3ed, #9f7aea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    margin-top: 10px;
    color: #718096;
    font-size: 0.95rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.15);
}

/* ── Input panels ── */
.panel {
    background: #0d1928;
    border: 1px solid rgba(99,179,237,0.10);
    border-radius: 14px;
    padding: 22px 20px 18px 20px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.panel:hover {
    border-color: rgba(99,179,237,0.25);
}

/* ── Streamlit widget overrides ── */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

div[data-baseweb="select"] > div {
    background-color: #111f2e !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 6px;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(90deg, #2b6cb0, #553c9a);
    color: #f0f6ff;
    border: none;
    border-radius: 10px;
    height: 3.2em;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    width: 100%;
    letter-spacing: 0.03em;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #3182ce, #6b46c1);
    box-shadow: 0 0 24px rgba(99,130,237,0.35);
    transform: translateY(-1px);
}

/* ── Result header ── */
.result-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63b3ed;
    margin: 28px 0 6px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.15);
}

/* ── Score card ── */
.score-card {
    background: #0d1928;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-top: 8px;
}
.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.score-label {
    font-size: 0.82rem;
    color: #718096;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 12px;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.04em;
}
.risk-low    { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.30); }
.risk-medium { background: rgba(236,201,75,0.15);  color: #f6e05e; border: 1px solid rgba(236,201,75,0.30); }
.risk-high   { background: rgba(245,101,101,0.15); color: #fc8181; border: 1px solid rgba(245,101,101,0.30); }

/* ── Factor row ── */
.factor-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
    color: #cbd5e0;
    line-height: 1.5;
}
.factor-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.factor-dot-high   { background: #fc8181; }
.factor-dot-medium { background: #f6e05e; }
.factor-dot-low    { background: #68d391; }

/* ── Retention card ── */
.retention-card {
    background: linear-gradient(135deg, #0d1f35, #1a103a);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 22px 20px;
    margin-top: 8px;
}
.retention-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9f7aea;
    margin-bottom: 14px;
}
.retention-action {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    font-size: 0.87rem;
    color: #a0aec0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.retention-action:last-child { border-bottom: none; }
.action-icon { font-size: 1rem; }

/* ── Sidebar model stat ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.82rem;
}
.stat-key   { color: #718096; }
.stat-value { color: #63b3ed; font-weight: 600; font-family: 'Space Grotesk', sans-serif; }

h1, h2, h3 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔮 ChurnSight")
    st.markdown("<p style='color:#718096;font-size:0.82rem;margin-top:-8px;'>Retention Intelligence Platform</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Model Performance**")
    for k, v in [("Algorithm", "Logistic Regression"),
                  ("ROC-AUC", "0.8403"),
                  ("Threshold", "0.40 (tuned)"),
                  ("CV Score", "0.843 ± 0.012"),
                  ("Training set", "5,634 customers"),
                  ("Class handling", "Balanced weights")]:
        st.markdown(f"""
        <div class='stat-row'>
            <span class='stat-key'>{k}</span>
            <span class='stat-value'>{v}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Key Churn Drivers**")
    for rank, driver in enumerate(["Tenure", "Monthly Charges", "Contract Type",
                                    "Total Charges", "Online Security"], 1):
        st.markdown(f"<p style='font-size:0.82rem;color:#a0aec0;margin:4px 0'>{rank}. {driver}</p>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:0.75rem;color:#4a5568;'>Built with Scikit-learn · SHAP · Streamlit<br>Dataset: IBM Telco Churn (7,043 customers)</p>",
                unsafe_allow_html=True)
    st.success("✅ Model Loaded")

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-eyebrow'>Retention Intelligence</div>
    <div class='hero-title'>📡 Telco Customer <span>Churn Predictor</span></div>
    <div class='hero-sub'>Enter customer details to predict churn risk and get actionable retention recommendations</div>
</div>
""", unsafe_allow_html=True)

# ── Input Grid ────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>👤 Customer Profile</div>", unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner", ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>🌐 Services</div>", unsafe_allow_html=True)
    internet_service  = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online Security",   ["Yes", "No", "No internet service"])
    online_backup     = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support      = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"])
    st.markdown("</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>📄 Contract</div>", unsafe_allow_html=True)
    contract          = st.selectbox("Contract Type",     ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>💳 Billing</div>", unsafe_allow_html=True)
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",   min_value=0.0, max_value=9000.0, value=500.0, step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────
if st.button("🚀 Predict Churn Risk", use_container_width=True):

    input_dict = {
        'gender':           1 if gender == "Male" else 0,
        'SeniorCitizen':    1 if senior == "Yes" else 0,
        'Partner':          1 if partner == "Yes" else 0,
        'Dependents':       1 if dependents == "Yes" else 0,
        'tenure':           tenure,
        'PhoneService':     1 if phone_service == "Yes" else 0,
        'MultipleLines':    {"No phone service": 0, "No": 1, "Yes": 2}[multiple_lines],
        'InternetService':  {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
        'OnlineSecurity':   {"No internet service": 0, "No": 1, "Yes": 2}[online_security],
        'OnlineBackup':     {"No internet service": 0, "No": 1, "Yes": 2}[online_backup],
        'DeviceProtection': {"No internet service": 0, "No": 1, "Yes": 2}[device_protection],
        'TechSupport':      {"No internet service": 0, "No": 1, "Yes": 2}[tech_support],
        'StreamingTV':      {"No internet service": 0, "No": 1, "Yes": 2}[streaming_tv],
        'StreamingMovies':  {"No internet service": 0, "No": 1, "Yes": 2}[streaming_movies],
        'Contract':         {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod':    {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        }[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges':   max(total_charges, monthly_charges),
    }

    input_df     = pd.DataFrame([input_dict])[feature_names].fillna(0)
    input_scaled = scaler.transform(input_df)
    prob         = model.predict_proba(input_scaled)[0][1]
    pct          = prob * 100

    # risk tier
    if prob < 0.30:
        tier, badge_cls, dot_cls, tier_color = "Low Risk",    "risk-low",    "factor-dot-low",    "#68d391"
    elif prob < 0.60:
        tier, badge_cls, dot_cls, tier_color = "Medium Risk", "risk-medium", "factor-dot-medium", "#f6e05e"
    else:
        tier, badge_cls, dot_cls, tier_color = "High Risk",   "risk-high",   "factor-dot-high",   "#fc8181"

    st.markdown("<div class='result-header'>Prediction Result</div>", unsafe_allow_html=True)

    res1, res2 = st.columns([1, 1], gap="large")

    # ── Gauge ─────────────────────────────────────────────────────
    with res1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': '%', 'font': {'size': 42, 'color': tier_color,
                                             'family': 'Space Grotesk'}},
            title={'text': "Churn Probability",
                   'font': {'size': 13, 'color': '#718096', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': '#4a5568', 'size': 11}},
                'bar': {'color': tier_color, 'thickness': 0.25},
                'bgcolor': '#0d1928',
                'bordercolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0, 30],  'color': 'rgba(72,187,120,0.12)'},
                    {'range': [30, 60], 'color': 'rgba(236,201,75,0.12)'},
                    {'range': [60, 100],'color': 'rgba(245,101,101,0.12)'},
                ],
                'threshold': {
                    'line': {'color': tier_color, 'width': 3},
                    'thickness': 0.85,
                    'value': pct
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter'},
            margin=dict(t=30, b=10, l=20, r=20),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style='text-align:center;margin-top:-10px'>
            <span class='risk-badge {badge_cls}'>{tier} — {pct:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    # ── Risk Factors + Retention ───────────────────────────────────
    with res2:
        st.markdown("<div class='section-label' style='margin-top:8px'>🔍 Risk Factors</div>",
                    unsafe_allow_html=True)

        factors = []
        if contract == "Month-to-month":
            factors.append(("high",   "Month-to-month contract — 42.7% industry churn rate vs 2.8% on two-year plans."))
        if tenure < 12:
            factors.append(("high",   f"Low tenure ({tenure} months) — median churner leaves at 10 months."))
        if tech_support == "No":
            factors.append(("medium", "No tech support — customers without it churn at 3× the rate."))
        if online_security == "No":
            factors.append(("medium", "No online security — 41.8% churn rate vs 14.6% with it."))
        if internet_service == "Fiber optic":
            factors.append(("medium", "Fiber optic service — historically elevated churn segment."))
        if monthly_charges > 75:
            factors.append(("medium", f"High monthly charges (${monthly_charges:.0f}) — churners average $74.44/mo."))
        if paperless_billing == "Yes":
            factors.append(("low",    "Paperless billing — mild positive correlation with churn."))

        if not factors:
            factors.append(("low", "No significant churn indicators detected for this customer profile."))

        for severity, text in factors:
            st.markdown(f"""
            <div class='factor-row'>
                <div class='factor-dot factor-dot-{severity}'></div>
                <div>{text}</div>
            </div>""", unsafe_allow_html=True)

        # ── Retention actions ──────────────────────────────────────
        st.markdown("<div class='retention-card' style='margin-top:16px'>", unsafe_allow_html=True)
        st.markdown("<div class='retention-title'>💡 Recommended Actions</div>", unsafe_allow_html=True)

        actions = []
        if prob >= 0.40:
            if contract == "Month-to-month":
                actions.append(("🎯", "Offer a loyalty discount for switching to a 1-year contract"))
            if tech_support == "No":
                actions.append(("🛠️", "Proactively offer a free tech support trial (30 days)"))
            if online_security == "No":
                actions.append(("🔒", "Bundle online security at reduced rate as retention offer"))
            if tenure < 12:
                actions.append(("🤝", "Assign a dedicated onboarding specialist to this account"))
            if monthly_charges > 75:
                actions.append(("💰", "Review pricing — customer may benefit from a plan downgrade"))
        else:
            actions.append(("✅", "No immediate action required — monitor at next billing cycle"))

        for icon, action in actions[:4]:
            st.markdown(f"""
            <div class='retention-action'>
                <span class='action-icon'>{icon}</span>
                <span>{action}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── What-If Analysis ──────────────────────────────────────────
    if prob >= 0.40:
        st.markdown("<div class='result-header' style='margin-top:24px'>⚡ What-If Analysis</div>",
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#718096;font-size:0.85rem;margin-bottom:16px'>How would retention interventions change this customer's risk?</p>",
                    unsafe_allow_html=True)

        scenarios = {
            "Switch to 1-year contract": dict(input_dict, Contract=1),
            "Switch to 2-year contract": dict(input_dict, Contract=2),
            "Add Tech Support":          dict(input_dict, TechSupport=2),
            "Add Online Security":       dict(input_dict, OnlineSecurity=2),
        }

        wi_cols = st.columns(len(scenarios))
        for col, (label, scenario) in zip(wi_cols, scenarios.items()):
            s_df      = pd.DataFrame([scenario])[feature_names].fillna(0)
            s_scaled  = scaler.transform(s_df)
            s_prob    = model.predict_proba(s_scaled)[0][1]
            delta     = s_prob - prob
            delta_str = f"{'↓' if delta < 0 else '↑'} {abs(delta*100):.1f}pp"
            d_color   = "#68d391" if delta < 0 else "#fc8181"
            with col:
                st.markdown(f"""
                <div style='background:#0d1928;border:1px solid rgba(99,179,237,0.10);
                            border-radius:12px;padding:16px;text-align:center'>
                    <div style='font-size:0.72rem;color:#718096;margin-bottom:8px;
                                text-transform:uppercase;letter-spacing:0.08em'>{label}</div>
                    <div style='font-family:Space Grotesk;font-size:1.8rem;font-weight:700;
                                color:#e2e8f0'>{s_prob*100:.1f}%</div>
                    <div style='font-size:0.85rem;color:{d_color};font-weight:600;
                                margin-top:4px'>{delta_str}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='color:#4a5568;font-size:0.75rem;margin-top:20px;text-align:center'>Model: Logistic Regression · Threshold: {THRESHOLD} · ROC-AUC: 0.8403</p>",
                unsafe_allow_html=True)