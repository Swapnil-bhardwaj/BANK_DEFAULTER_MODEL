import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# BANK LOAN DEFAULT PREDICTION WEB APPLICATION
# ============================================================

MODEL_FILE = "bank_loan_dataset_output/ml/loan_default_model.pkl"
DATA_FILE = "bank_loan_dataset_output/bank_loans_cleaned.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #0A2540;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #5f6b7a;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .default-box {
        background-color: #ffe5e5;
        border: 2px solid #d93025;
    }

    .safe-box {
        background-color: #e4f7ec;
        border: 2px solid #18864b;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

    .probability {
        font-size: 22px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        "Trained model not found. "
        "Please run ml_model.py first."
    )

    st.stop()


model = joblib.load(MODEL_FILE)


# ============================================================
# LOAD DATASET
# ============================================================

if os.path.exists(DATA_FILE):

    df = pd.read_csv(DATA_FILE)

else:

    df = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏦 Bank Loan Default Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Synthetic Loan Risk Analysis System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Loan Prediction",
        "Dataset Overview",
        "Model Performance",
        "Feature Importance"
    ]
)


# ============================================================
# PAGE 1 — LOAN PREDICTION
# ============================================================

if page == "Loan Prediction":

    st.header("📋 Customer Loan Information")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    with col1:

        st.subheader("Personal Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=35
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced",
                "Widowed"
            ]
        )

        education = st.selectbox(
            "Education",
            [
                "High School",
                "Graduate",
                "Postgraduate",
                "Other"
            ]
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=5,
            value=1
        )

        residence_type = st.selectbox(
            "Residence Type",
            [
                "Owned",
                "Rented",
                "Family"
            ]
        )

    # --------------------------------------------------------
    # EMPLOYMENT INFORMATION
    # --------------------------------------------------------

    with col2:

        st.subheader("Employment & Income")

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Salaried",
                "Self-Employed",
                "Business Owner",
                "Unemployed",
                "Retired"
            ]
        )

        employment_years = st.number_input(
            "Employment Years",
            min_value=0.0,
            max_value=40.0,
            value=8.0
        )

        annual_income = st.number_input(
            "Annual Income (₹)",
            min_value=0.0,
            value=600000.0,
            step=10000.0
        )

        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=850.0,
            value=700.0
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=15,
            value=1
        )

    # --------------------------------------------------------
    # CREDIT & LOAN INFORMATION
    # --------------------------------------------------------

    with col3:

        st.subheader("Credit & Loan Details")

        previous_defaults = st.number_input(
            "Previous Defaults",
            min_value=0,
            max_value=10,
            value=0
        )

        late_payments_12m = st.number_input(
            "Late Payments (Last 12 Months)",
            min_value=0,
            max_value=12,
            value=0
        )

        credit_utilization = st.number_input(
            "Credit Utilization",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01
        )

        months_since_last_delinquency = st.number_input(
            "Months Since Last Delinquency",
            min_value=0,
            max_value=60,
            value=60
        )

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

        loan_term_months = st.number_input(
            "Loan Term (Months)",
            min_value=6,
            max_value=84,
            value=48
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=45.0,
            value=12.0,
            step=0.1
        )

    with col5:

        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=0.9,
            value=0.40,
            step=0.01
        )

    with col6:

        collateral_value = st.number_input(
            "Collateral Value (₹)",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Personal",
            "Home",
            "Auto",
            "Education",
            "Business",
            "Medical"
        ]
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation_errors = []

    expected_monthly_income = annual_income / 12

    if annual_income > 0:

        difference = abs(
            monthly_income - expected_monthly_income
        )

        if difference > expected_monthly_income * 0.20:

            validation_errors.append(
                "Annual income and monthly income "
                "appear inconsistent."
            )

    if employment_years > age - 16:

        validation_errors.append(
            "Employment years appear inconsistent with age."
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("---")

    predict_button = st.button(
        "🔍 Predict Loan Default",
        use_container_width=True
    )


    if predict_button:

        if validation_errors:

            st.warning("Please review the following:")

            for error in validation_errors:

                st.write("⚠️", error)

        else:

            # Create input dataframe
            customer = pd.DataFrame([{

                "age": age,
                "gender": gender,
                "marital_status": marital_status,
                "education": education,
                "dependents": dependents,
                "residence_type": residence_type,
                "employment_type": employment_type,
                "employment_years": employment_years,
                "annual_income": annual_income,
                "monthly_income": monthly_income,
                "credit_score": credit_score,
                "existing_loans": existing_loans,
                "previous_defaults": previous_defaults,
                "late_payments_12m": late_payments_12m,
                "credit_utilization": credit_utilization,
                "months_since_last_delinquency":
                    months_since_last_delinquency,
                "loan_amount": loan_amount,
                "loan_term_months": loan_term_months,
                "interest_rate": interest_rate,
                "dti_ratio": dti_ratio,
                "collateral_value": collateral_value,
                "loan_purpose": loan_purpose,

                "application_year": 2026,
                "application_month": 9,
                "application_day": 28
            }])


            # Prediction
            prediction = model.predict(customer)[0]

            probability = model.predict_proba(customer)[0]

            non_default_probability = probability[0] * 100
            default_probability = probability[1] * 100


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.subheader("📊 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:

                    st.error(
                        "⚠️ Prediction: LOAN DEFAULT"
                    )

                else:

                    st.success(
                        "✅ Prediction: NON-DEFAULT"
                    )

            with result_col2:

                st.metric(
                    "Default Probability",
                    f"{default_probability:.2f}%"
                )

            st.progress(
                int(default_probability)
            )

            st.info(
                "This prediction is generated by a "
                "machine-learning model trained on a "
                "synthetic educational dataset. It should "
                "not be used as an actual lending decision."
            )


# ============================================================
# PAGE 2 — DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    if df is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Records",
                f"{len(df):,}"
            )

        with col2:

            st.metric(
                "Features",
                len(df.columns) - 1
            )

        with col3:

            default_rate = (
                df["loan_default"].mean() * 100
            )

            st.metric(
                "Default Rate",
                f"{default_rate:.2f}%"
            )

        with col4:

            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )


        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Loan Default Distribution")

        default_counts = (
            df["loan_default"]
            .value_counts()
            .rename(
                {
                    0: "Non-Default",
                    1: "Default"
                }
            )
        )

        st.bar_chart(default_counts)


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🤖 Model Performance")

    comparison_file = (
        "bank_loan_dataset_output/ml/"
        "model_comparison.csv"
    )

    if os.path.exists(comparison_file):

        results = pd.read_csv(
            comparison_file
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        st.subheader("Model Comparison")

        chart_data = results.set_index(
            "model"
        )[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc"
            ]
        ]

        st.bar_chart(chart_data)

    else:

        st.warning(
            "Model comparison file not found."
        )


# ============================================================
# PAGE 4 — FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.header("📈 Feature Importance")

    importance_file = (
        "bank_loan_dataset_output/ml/"
        "feature_importance.csv"
    )

    if os.path.exists(importance_file):

        importance = pd.read_csv(
            importance_file
        )

        top_features = (
            importance
            .head(15)
            .sort_values(
                "importance"
            )
        )

        st.bar_chart(
            top_features.set_index(
                "feature"
            )
        )

        st.subheader(
            "Top Predictive Features"
        )

        st.dataframe(
            importance.head(15),
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance file not found."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Bank Loan Default Prediction System | "
    "Educational Synthetic Dataset | "
    "Machine Learning Project"
)