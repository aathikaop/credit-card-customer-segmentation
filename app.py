# Import Libraries

import streamlit as st
import pandas as pd
import joblib

from src.feature_engineering import add_features


# ------------------------------------------------------------------
# Load the fitted preprocessing pipeline and the trained clustering model
# ------------------------------------------------------------------

PREPROCESSING_PATH = "outputs/models/preprocessing_pipeline.pkl"
MODEL_PATH = "outputs/models/best_clustering_model.pkl"

preprocessing_pipeline = joblib.load(PREPROCESSING_PATH)
model = joblib.load(MODEL_PATH)


CLUSTER_INFO = {
    0: {
        "label": "Cash Advance Revolvers",
        "description": "High cash advance usage, high balance, rarely pays in full. "
                        "High interest revenue for the bank, but higher credit risk."
    },
    1: {
        "label": "Low Engagement / Inactive",
        "description": "Low balance, low spending, low activity overall. "
                        "Low value, good candidate for re-engagement campaigns."
    },
    2: {
        "label": "Responsible High Spenders",
        "description": "Highest purchases and highest full-payment rate. "
                        "Low risk, high value - ideal customers."
    },
    3: {
        "label": "Cash-Only Users",
        "description": "Rarely makes purchases but relies heavily on cash advances. "
                        "Uses the card more like a loan than for shopping."
    }
}


# ------------------------------------------------------------------
# Data-driven lookup tables for the fields a customer can't reasonably
# know exactly (frequencies, transaction counts). These were computed
# by grouping the real training data by cash-advance usage level and
# full-payment habit, then taking the median of each hidden field
# within each group - not arbitrary guesses.
# ------------------------------------------------------------------

CASH_ADVANCE_LOOKUP = {
    "Never":        {"CASH_ADVANCE": 0.0,    "CASH_ADVANCE_FREQUENCY": 0.0,   "CASH_ADVANCE_TRX": 0},
    "Occasionally": {"CASH_ADVANCE": 759.0,  "CASH_ADVANCE_FREQUENCY": 0.167, "CASH_ADVANCE_TRX": 2},
    "Frequently":   {"CASH_ADVANCE": 2337.84,"CASH_ADVANCE_FREQUENCY": 0.417, "CASH_ADVANCE_TRX": 10},
}

FULL_PAYMENT_LOOKUP = {
    "Never":     {"PRC_FULL_PAYMENT": 0.0,   "PAYMENTS": 714.21,  "MINIMUM_PAYMENTS": 512.03},
    "Rarely":    {"PRC_FULL_PAYMENT": 0.091, "PAYMENTS": 1858.57, "MINIMUM_PAYMENTS": 276.95},
    "Sometimes": {"PRC_FULL_PAYMENT": 0.333, "PAYMENTS": 987.79,  "MINIMUM_PAYMENTS": 156.42},
    "Always":    {"PRC_FULL_PAYMENT": 0.917, "PAYMENTS": 1112.76, "MINIMUM_PAYMENTS": 165.12},
}

# Overall dataset medians - used for fields not asked about at all
# (these have little influence on the segment and aren't something
# a customer would know offhand)
DEFAULTS = {
    "BALANCE_FREQUENCY": 1.0,
    "PURCHASES_FREQUENCY": 0.5,
    "ONEOFF_PURCHASES_FREQUENCY": 0.0833,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.1667,
    "PURCHASES_TRX": 7,
}


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------

st.set_page_config(page_title="Customer Segment Predictor", layout="centered")

st.title("Credit Card Customer Segment Predictor")
st.write(
    "Answer a few simple questions about your card usage to see which "
    "customer segment you belong to."
)


# ------------------------------------------------------------------
# Simple input form - only things a customer could realistically answer
# ------------------------------------------------------------------

with st.form("customer_form"):

    balance = st.number_input(
        "Roughly what balance do you usually carry on your card?",
        min_value=0.0, value=800.0
    )

    credit_limit = st.number_input(
        "What is your credit limit?",
        min_value=0.0, value=3000.0
    )

    monthly_purchase = st.number_input(
        "Roughly how much do you spend on purchases per month?",
        min_value=0.0, value=350.0
    )

    purchase_style = st.slider(
        "Do you mostly buy in one go, or in installments?",
        0, 100, 50,
        help="0 = always installments, 100 = always one-time purchases"
    )

    cash_advance_usage = st.selectbox(
        "How often do you take a cash advance on this card?",
        ["Never", "Occasionally", "Frequently"]
    )

    full_payment_habit = st.selectbox(
        "How often do you pay your full bill (not just the minimum)?",
        ["Never", "Rarely", "Sometimes", "Always"]
    )

    tenure = st.slider(
        "How many months have you had this card?",
        6, 12, 12
    )

    submitted = st.form_submit_button("Predict My Segment")


# ------------------------------------------------------------------
# On submit: fill in the full 17-column feature set using the direct
# answers + the data-driven lookups, then run the same pipeline used
# during training
# ------------------------------------------------------------------

if submitted:

    cash = CASH_ADVANCE_LOOKUP[cash_advance_usage]
    full_pay = FULL_PAYMENT_LOOKUP[full_payment_habit]

    oneoff_share = purchase_style / 100
    oneoff_purchases = monthly_purchase * oneoff_share
    installments_purchases = monthly_purchase * (1 - oneoff_share)

    new_customer = pd.DataFrame([{
        "BALANCE": balance,
        "BALANCE_FREQUENCY": DEFAULTS["BALANCE_FREQUENCY"],
        "PURCHASES": monthly_purchase,
        "ONEOFF_PURCHASES": oneoff_purchases,
        "INSTALLMENTS_PURCHASES": installments_purchases,
        "CASH_ADVANCE": cash["CASH_ADVANCE"],
        "PURCHASES_FREQUENCY": DEFAULTS["PURCHASES_FREQUENCY"],
        "ONEOFF_PURCHASES_FREQUENCY": DEFAULTS["ONEOFF_PURCHASES_FREQUENCY"],
        "PURCHASES_INSTALLMENTS_FREQUENCY": DEFAULTS["PURCHASES_INSTALLMENTS_FREQUENCY"],
        "CASH_ADVANCE_FREQUENCY": cash["CASH_ADVANCE_FREQUENCY"],
        "CASH_ADVANCE_TRX": cash["CASH_ADVANCE_TRX"],
        "PURCHASES_TRX": DEFAULTS["PURCHASES_TRX"],
        "CREDIT_LIMIT": credit_limit,
        "PAYMENTS": full_pay["PAYMENTS"],
        "MINIMUM_PAYMENTS": full_pay["MINIMUM_PAYMENTS"],
        "PRC_FULL_PAYMENT": full_pay["PRC_FULL_PAYMENT"],
        "TENURE": tenure
    }])

    new_customer_features = add_features(new_customer)

    new_customer_processed = preprocessing_pipeline.transform(new_customer_features)
    new_customer_processed = pd.DataFrame(
        new_customer_processed,
        columns=new_customer_features.drop(columns=["CUST_ID"], errors="ignore").columns
    )

    if hasattr(model, "predict"):
        cluster = model.predict(new_customer_processed)[0]
    else:
        st.error(
            "The saved model does not support predicting on new data. "
            "Re-run the pipeline and make sure K-Means or GMM is selected as the best model."
        )
        st.stop()

    info = CLUSTER_INFO.get(cluster, {"label": f"Cluster {cluster}", "description": "No description available."})

    st.success(f"Predicted Segment: **{info['label']}** (Cluster {cluster})")
    st.write(info["description"])