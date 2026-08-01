import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Predictor", layout="centered")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center;'>🏦 Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict loan approval using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")
columns = joblib.load("model_columns.pkl")

# ---------------- INPUT UI ----------------
st.subheader("📋 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("### 💰 Financial Details")

coapp_income = st.slider("Coapplicant Income", 0, 10000, 0)
app_income = st.slider("Applicant Income", 0, 20000, 5000)
loan_amount = st.slider("Loan Amount (in thousands)", 0, 500, 120)
loan_term = st.slider("Loan Term", 0, 600, 360)

# ---------------- ENCODING ----------------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0

if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

# ---------------- TRANSFORM ----------------
total_income = app_income + coapp_income

app_income_log = np.log(app_income + 1)
coapp_income_log = np.log(coapp_income + 1)
loan_amount_log = np.log(loan_amount + 1)
loan_term_log = np.log(loan_term + 1)

# ---------------- BUTTON ----------------
if st.button("🚀 Predict Loan Status"):

    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Amount_Term": loan_term_log,
        "CoapplicantIncome": coapp_income_log,
        "ApplicantIncome": app_income_log,
        "LoanAmount": loan_amount_log
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # ---------------- PREDICT ----------------
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)
    approval_prob = prob[0][1] * 100

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.progress(int(approval_prob))
    st.write(f"Approval Probability: **{approval_prob:.2f}%**")

    # ---------------- GRAPH ----------------
    st.subheader("📈 Income vs Loan Analysis")

    fig, ax = plt.subplots()

    x = ["Applicant Income", "Coapplicant Income", "Loan Amount"]
    y = [app_income, coapp_income, loan_amount]

    ax.bar(x, y)
    ax.set_ylabel("Amount")
    ax.set_title("Financial Overview")

    st.pyplot(fig)

    # ---------------- EXTRA INSIGHT ----------------
    st.subheader("💡 Insight")

    if approval_prob > 70:
        st.info("High chances of loan approval 👍")
    elif approval_prob > 40:
        st.warning("Moderate chances — improve credit/income")
    else:
        st.error("Low chances — try improving financial profile")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made by Tarushi Modi 💻</p>", unsafe_allow_html=True)