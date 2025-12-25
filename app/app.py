import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Employee Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# Load Model Artifacts
# --------------------------------------------------
artifact = joblib.load("Notebooks/churn_model.pkl")
model = artifact["model"]
scaler = artifact["scaler"]
THRESHOLD = artifact["threshold"]

# --------------------------------------------------
# Custom CSS (UI + Color Grading)
# --------------------------------------------------
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Main Card */
.main-card {
    background: #020617;
    padding: 2.5rem;
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    border: 1px solid #1e293b;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    color: #e5e7eb;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-size: 16px;
}

/* Section Title */
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #e5e7eb;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-left: 5px solid #38bdf8;
    padding-left: 12px;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #1e293b, transparent);
    margin: 2rem 0;
}

/* Inputs */
label {
    color: #cbd5f5 !important;
    font-weight: 500;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #2563eb);
    color: #020617;
    border-radius: 10px;
    height: 3.2rem;
    font-size: 18px;
    font-weight: 600;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #2563eb, #38bdf8);
}

/* Result Card */
.result-card {
    background: #020617;
    padding: 2rem;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-top: 2rem;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-size: 32px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# UI Layout
# --------------------------------------------------

st.markdown('<div class="title">🎯 Employee Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to assess employee retention risk</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Personal Information
# --------------------------------------------------
st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

with col3:
    partner = st.selectbox("Has Partner", ["No", "Yes"])

with col4:
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Account Details
# --------------------------------------------------
st.markdown('<div class="section-title">💼 Account Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=5.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "Long Term"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

with col3:
    payment = st.selectbox("Payment Method", ["Electronic Check", "Other"])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Services
# --------------------------------------------------
st.markdown('<div class="section-title">🌐 Services & Features</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])

with col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Encode Input
# --------------------------------------------------
input_data = np.array([
    1 if gender == "Male" else 0,
    1 if senior == "Yes" else 0,
    1 if partner == "Yes" else 0,
    1 if dependents == "Yes" else 0,
    tenure,
    1 if phone == "Yes" else 0,
    1 if multiple == "Yes" else 0,
    0 if internet == "DSL" else 1 if internet == "Fiber Optic" else 2,
    1 if online_security == "Yes" else 0,
    1 if online_backup == "Yes" else 0,
    1 if device_protection == "Yes" else 0,
    1 if tech_support == "Yes" else 0,
    1 if streaming_tv == "Yes" else 0,
    1 if streaming_movies == "Yes" else 0,
    0 if contract == "Month-to-month" else 1,
    1 if paperless == "Yes" else 0,
    0 if payment == "Electronic Check" else 1,
    monthly
]).reshape(1, -1)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Analyze Churn Risk"):
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0, 1]
    prediction = int(prob >= THRESHOLD)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.metric("Churn Probability", f"{prob:.1%}")

    if prediction == 1:
        st.error("⚠️ High Risk — Immediate retention action recommended")
    else:
        st.success("✅ Low Risk — Employee likely to stay")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
