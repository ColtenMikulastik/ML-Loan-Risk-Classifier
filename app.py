
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("Credit Risk Prediction")
st.write(
    "Enter the applicant's information to estimate "
    "credit risk using machine learning."
)

# Select the machine learning model
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree"]
)

model_file = (
    "logistic_regression.pkl"
    if model_choice == "Logistic Regression"
    else "decision_tree.pkl"
)

model_path = Path("models") / model_file

if not model_path.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

# Convert user-friendly selections to dataset codes
checking_options = {
    "Less than 0 DM": "A11",
    "0 to 200 DM": "A12",
    "200 DM or more": "A13",
    "No checking account": "A14"
}

credit_history_options = {
    "No previous credit": "A30",
    "All previous credit paid on time": "A31",
    "Existing credit paid on time": "A32",
    "Previous payment delays": "A33",
    "Serious credit problems": "A34"
}

purpose_options = {
    "New car": "A40",
    "Used car": "A41",
    "Furniture or equipment": "A42",
    "Radio or television": "A43",
    "Appliances": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Retraining": "A48",
    "Business": "A49",
    "Other": "A410"
}

savings_options = {
    "Less than 100 DM": "A61",
    "100 to 500 DM": "A62",
    "500 to 1,000 DM": "A63",
    "1,000 DM or more": "A64",
    "Unknown or no savings account": "A65"
}

employment_options = {
    "Unemployed": "A71",
    "Less than 1 year": "A72",
    "1 to 4 years": "A73",
    "4 to 7 years": "A74",
    "7 years or more": "A75"
}

housing_options = {
    "Rent": "A151",
    "Own": "A152",
    "Free housing": "A153"
}

st.subheader("Applicant Information")

checking = st.selectbox(
    "Checking Account Balance",
    list(checking_options)
)

duration = st.number_input(
    "Loan Duration (Months)",
    min_value=1, max_value=120, value=24
)

credit_history = st.selectbox(
    "Credit History",
    list(credit_history_options),
    index=2
)

purpose = st.selectbox(
    "Loan Purpose",
    list(purpose_options)
)

credit_amount = st.number_input(
    "Loan Amount (DM)",
    min_value=1, max_value=100000, value=5000
)

savings = st.selectbox(
    "Savings Balance",
    list(savings_options)
)

employment = st.selectbox(
    "Employment Duration",
    list(employment_options),
    index=2
)

installment_rate = st.slider(
    "Installment Rate (% of Disposable Income)",
    min_value=1, max_value=4, value=2
)

age = st.number_input(
    "Applicant Age",
    min_value=18, max_value=100, value=35
)

housing = st.selectbox(
    "Housing Status",
    list(housing_options),
    index=1
)

# Generate prediction
if st.button("Predict Credit Risk", type="primary"):

    applicant = pd.DataFrame([{
        "checking_account": checking_options[checking],
        "duration": duration,
        "credit_history": credit_history_options[credit_history],
        "purpose": purpose_options[purpose],
        "credit_amount": credit_amount,
        "savings": savings_options[savings],
        "employment": employment_options[employment],
        "installment_rate": installment_rate,
        "age": age,
        "housing": housing_options[housing]
    }])

    prediction = model.predict(applicant)[0]
    probability = model.predict_proba(applicant)[0][1]

    st.subheader("Prediction Results")

    if prediction == 1:
        st.error("Prediction: High Credit Risk")
    else:
        st.success("Prediction: Low Credit Risk")

    st.metric(
        "Estimated Probability of High Risk",
        f"{probability * 100:.1f}%"
    )

    st.caption(
        f"Prediction generated using {model_choice}. "
        "This is an educational model trained on historical "
        "German credit data, not a lending decision."
    )
