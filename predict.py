import os
import joblib
import pandas as pd


# ============================================================
# LOAN DEFAULT PREDICTION
# ============================================================

MODEL_FILE = "bank_loan_dataset_output/ml/loan_default_model.pkl"


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):
    print("ERROR: Trained model not found.")
    print("Please run: python ml_model.py")
    exit()

model = joblib.load(MODEL_FILE)

print("=" * 60)
print("BANK LOAN DEFAULT PREDICTION SYSTEM")
print("=" * 60)


# ============================================================
# 2. GET CUSTOMER INFORMATION
# ============================================================

def get_float(prompt, minimum=None, maximum=None):

    while True:

        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Value must be >= {minimum}")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must be <= {maximum}")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


def get_int(prompt, minimum=None, maximum=None):

    while True:

        try:
            value = int(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Value must be >= {minimum}")
                continue

            if maximum is not None and value > maximum:
                print(f"Value must be <= {maximum}")
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def get_choice(prompt, choices):

    while True:

        value = input(
            f"{prompt} ({'/'.join(choices)}): "
        ).strip()

        # Case-insensitive matching
        for choice in choices:

            if value.lower() == choice.lower():
                return choice

        print("Please select one of:", ", ".join(choices))


print("\nEnter Customer Information")
print("-" * 60)


# ============================================================
# 3. CUSTOMER DETAILS
# ============================================================

age = get_float(
    "Age: ",
    18,
    70
)

gender = get_choice(
    "Gender",
    ["Male", "Female", "Other"]
)

marital_status = get_choice(
    "Marital Status",
    ["Single", "Married", "Divorced", "Widowed"]
)

education = get_choice(
    "Education",
    ["High School", "Graduate", "Postgraduate", "Other"]
)

dependents = get_int(
    "Number of Dependents: ",
    0,
    5
)

residence_type = get_choice(
    "Residence Type",
    ["Owned", "Rented", "Family"]
)

employment_type = get_choice(
    "Employment Type",
    [
        "Salaried",
        "Self-Employed",
        "Business Owner",
        "Unemployed",
        "Retired"
    ]
)

employment_years = get_float(
    "Employment Years: ",
    0,
    40
)

annual_income = get_float(
    "Annual Income (₹): ",
    0
)

monthly_income = get_float(
    "Monthly Income (₹): ",
    0
)

credit_score = get_float(
    "Credit Score: ",
    300,
    850
)

existing_loans = get_int(
    "Existing Loans: ",
    0,
    15
)

previous_defaults = get_int(
    "Previous Defaults: ",
    0,
    10
)

late_payments_12m = get_int(
    "Late Payments in Last 12 Months: ",
    0,
    12
)

credit_utilization = get_float(
    "Credit Utilization (0 to 1): ",
    0,
    1
)

months_since_last_delinquency = get_int(
    "Months Since Last Delinquency: ",
    0,
    60
)

loan_amount = get_float(
    "Loan Amount (₹): ",
    0
)

loan_term_months = get_int(
    "Loan Term (Months): ",
    6,
    84
)

interest_rate = get_float(
    "Interest Rate (%): ",
    0,
    45
)

dti_ratio = get_float(
    "DTI Ratio (0 to 0.9): ",
    0,
    0.9
)

collateral_value = get_float(
    "Collateral Value (₹): ",
    0
)

loan_purpose = get_choice(
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


# ============================================================
# 4. CREATE INPUT DATAFRAME
# ============================================================

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
    "months_since_last_delinquency": months_since_last_delinquency,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "interest_rate": interest_rate,
    "dti_ratio": dti_ratio,
    "collateral_value": collateral_value,
    "loan_purpose": loan_purpose,

    # Same date-derived features used during training
    "application_year": 2026,
    "application_month": 9,
    "application_day": 28
}])


# ============================================================
# 5. MAKE PREDICTION
# ============================================================

prediction = model.predict(customer)[0]

probability = model.predict_proba(customer)[0]

non_default_probability = probability[0] * 100
default_probability = probability[1] * 100


# ============================================================
# 6. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

if prediction == 1:

    print("Prediction: LOAN DEFAULT")

else:

    print("Prediction: NON-DEFAULT")


print(
    f"Default Probability: "
    f"{default_probability:.2f}%"
)

print(
    f"Non-Default Probability: "
    f"{non_default_probability:.2f}%"
)

print("=" * 60)

print("\nPrediction completed.")