import streamlit as st

from churn.model import (
    MODEL_FACTS,
    THRESHOLD,
    encode,
    load_artifacts,
    predict,
    retention_actions,
    risk_factors,
    risk_tier,
    what_if,
)

st.set_page_config(page_title="ChurnSight", page_icon=":material/query_stats:", layout="wide")

art = st.cache_resource(load_artifacts)()

with st.sidebar:
    st.subheader("About the model")
    for k, v in MODEL_FACTS.items():
        st.markdown(f"**{k}**  \n{v}")
    st.caption("Trained on the IBM Telco dataset (7,043 customers). Built by Mustafa Amjad.")
    st.link_button(
        "Source code",
        "https://github.com/mustafa-amjad/telco-churn-prediction",
        icon=":material/code:",
        width="stretch",
    )

st.title("ChurnSight")
st.caption("Score a telecom customer's churn risk, see what drives it, and test what would lower it.")

with st.form("customer"):
    profile, services = st.columns(2, gap="large")
    with profile:
        st.subheader("Customer")
        gender = st.segmented_control("Gender", ["Male", "Female"], default="Male")
        senior = st.segmented_control("Senior citizen", ["No", "Yes"], default="No")
        partner = st.segmented_control("Has a partner", ["Yes", "No"], default="No")
        dependents = st.segmented_control("Has dependents", ["Yes", "No"], default="No")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        st.subheader("Contract and billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.segmented_control("Paperless billing", ["Yes", "No"], default="Yes")
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        bills = st.columns(2)
        monthly_charges = bills[0].number_input("Monthly charges ($)", 0.0, 150.0, 65.0, 0.5)
        total_charges = bills[1].number_input("Total charges ($)", 0.0, 9000.0, 500.0, 1.0)
    with services:
        st.subheader("Services")
        phone_service = st.segmented_control("Phone service", ["Yes", "No"], default="Yes")
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        addon = ["Yes", "No", "No internet service"]
        online_security = st.selectbox("Online security", addon, index=1)
        online_backup = st.selectbox("Online backup", addon, index=1)
        device_protection = st.selectbox("Device protection", addon, index=1)
        tech_support = st.selectbox("Tech support", addon, index=1)
        streaming_tv = st.selectbox("Streaming TV", addon, index=1)
        streaming_movies = st.selectbox("Streaming movies", addon, index=1)
    submitted = st.form_submit_button("Score this customer", type="primary", icon=":material/insights:")

customer = dict(
    gender=gender or "Male", senior=senior or "No", partner=partner or "No",
    dependents=dependents or "No", tenure=tenure, phone_service=phone_service or "Yes",
    multiple_lines=multiple_lines, internet_service=internet_service,
    online_security=online_security, online_backup=online_backup,
    device_protection=device_protection, tech_support=tech_support,
    streaming_tv=streaming_tv, streaming_movies=streaming_movies, contract=contract,
    paperless_billing=paperless_billing or "Yes", payment_method=payment_method,
    monthly_charges=monthly_charges, total_charges=total_charges,
)

if submitted:
    st.session_state.scored = customer

if "scored" not in st.session_state:
    st.caption("Fill in the customer details and select **Score this customer**.")
    st.stop()

c = st.session_state.scored
features = encode(c)
prob = predict(art, features)
tier, color = risk_tier(prob)

st.header("Result")
left, right = st.columns([1, 2], gap="large")
with left:
    with st.container(border=True):
        st.metric("Churn probability", f"{prob:.0%}")
        st.badge(tier, color=color)
        st.progress(prob)
        flagged = prob >= THRESHOLD
        st.caption(
            f"{'Flagged' if flagged else 'Not flagged'} for retention "
            f"(threshold {THRESHOLD:.2f})."
        )
with right:
    st.subheader("What drives this score")
    factors = risk_factors(c)
    if factors:
        badge = {"High": ":red-badge[High]", "Medium": ":orange-badge[Medium]", "Low": ":gray-badge[Low]"}
        st.markdown("\n".join(f"- {badge[w]} {s}" for w, s in factors))
    else:
        st.caption("No strong churn signals in this profile.")

    actions = retention_actions(c, prob)
    if actions:
        st.subheader("Suggested retention actions")
        st.markdown("\n".join(f"- {a}" for a in actions))

if prob >= THRESHOLD:
    st.subheader("What if we changed one thing?")
    st.caption("Churn probability if the customer accepted each retention offer.")
    grid = st.columns(2)
    for i, (name, p) in enumerate(what_if(art, features)):
        grid[i % 2].metric(
            name, f"{p:.0%}", delta=f"{(p - prob) * 100:+.1f} pts", delta_color="inverse", border=True
        )
