import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Prediction System", page_icon="🏦", layout="centered")

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Prediction System")
st.markdown("Fill in the applicant details below to check **loan approval eligibility**.")
st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("loan_form"):

    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col3:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    col4, col5 = st.columns(2)
    with col4:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
    with col5:
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

    st.divider()
    st.subheader("💼 Employment & Income")
    col6, col7 = st.columns(2)
    with col6:
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
        employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
    with col7:
        applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=50000, step=1000)
        coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, value=0, step=1000)

    monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=20000, step=500)

    st.divider()
    st.subheader("🏠 Property & Loan Details")
    col8, col9 = st.columns(2)
    with col8:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        property_value = st.number_input("Property Value (₹)", min_value=0, value=2000000, step=50000)
    with col9:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=500000, step=10000)
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=120, step=12)

    st.divider()
    st.subheader("📊 Credit Profile")
    col10, col11, col12 = st.columns(3)
    with col10:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    with col11:
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    with col12:
        previous_loans = st.number_input("Previous Loans", min_value=0, max_value=20, value=0)

    st.divider()
    submitted = st.form_submit_button("🔍 Predict Loan Eligibility", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Encode categorical features (same order as training)
    le = LabelEncoder()

    gender_enc = 1 if gender == "Male" else 0

    marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
    marital_enc = marital_map[marital_status]

    edu_map = {"Graduate": 0, "Not Graduate": 1}
    edu_enc = edu_map[education]

    emp_map = {"Employed": 0, "Self-Employed": 2, "Unemployed": 1}
    emp_enc = emp_map[employment_status]

    area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    area_enc = area_map[property_area]

    # Build feature array (same column order as training)
    features = np.array([[
        credit_score, credit_history, applicant_income,
        coapplicant_income, loan_amount, loan_term,
        employment_years, age, dependents, previous_loans,
        monthly_expenses, property_value,
        gender_enc, marital_enc, edu_enc,
        emp_enc, area_enc
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.divider()
    st.subheader("📋 Prediction Result")

    if prediction == 0:
        st.success("✅ **Loan Approved!**  The applicant is likely to repay the loan.")
        prob_pct = probability[0] * 100
    else:
        st.error("❌ **Loan Rejected.**  The applicant is flagged as high risk.")
        prob_pct = probability[1] * 100

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Confidence", f"{prob_pct:.1f}%")
    with col_b:
        risk_label = "High Risk 🔴" if prediction == 1 else "Low Risk 🟢"
        st.metric("Risk Level", risk_label)

    st.progress(int(probability[0] * 100), text=f"Approval probability: {probability[0]*100:.1f}%")