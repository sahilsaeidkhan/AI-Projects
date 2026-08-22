import streamlit as st 
import pandas as pd 

from loan_model import LoanApprovalApp

st.set_page_config( page_title="Loan Approval Predictor")

st.title("Loan Approval Prediction")

st.write("Enter applicant details to check the loan eligibility.")

@st.cache_resource
def load_model():
    return LoanApprovalApp()

model = load_model()

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0.0
)

loan_amount = st.number_input(
    "Loan Amount Requested",
    min_value=0.0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=1
)

cibil_score = st.number_input(
    "CIBIL Score"
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0.0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0.0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0.0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0.0
)

if st.button("Predict"):

    data = {

        "no_of_dependents": no_of_dependents,

        "education": education,

        "self_employed": self_employed,

        "income_annum": income_annum,

        "loan_amount": loan_amount,

        "loan_term": loan_term,

        "cibil_score": cibil_score,

        "residential_assets_value":
            residential_assets_value,

        "commercial_assets_value":
            commercial_assets_value,

        "luxury_assets_value":
            luxury_assets_value,

        "bank_asset_value":
            bank_asset_value
    }  

    applicant_df = pd.DataFrame([data])

    result = model.two_stage_predict(applicant_df)

    if result["approve"] == 1:

        st.success("Loan Approved")

        st.metric("Predicted Loan Amount" , 
                  f"{result['regression_prediction']:,.2f}")

    else:

        st.error("Loan Rejected ")
