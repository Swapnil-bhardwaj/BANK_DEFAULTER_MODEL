
"""
BANK LOAN DATASET GENERATOR
---------------------------
Generates a fully synthetic banking loan dataset using only Python code.

Outputs:
    bank_loans_clean.csv   -> clean synthetic dataset
    bank_loans_raw.csv     -> same dataset with intentional anomalies
    anomaly_log.csv        -> records exactly what anomalies were injected
    data_dictionary.csv    -> feature definitions
    dataset_summary.txt    -> generation summary

Install:
    pip install pandas numpy

Run:
    python bank_loan_dataset_generator.py

Important:
    This is synthetic educational data. It is NOT real bank/customer data.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
N_ROWS = 20_000

OUTPUT_DIR = Path("bank_loan_dataset_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
random.seed(SEED)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def choice(values, probabilities, size):
    return rng.choice(values, size=size, p=probabilities)


def add_log(log, row_index, customer_id, column,
            anomaly_type, original_value, corrupted_value):
    log.append({
        "row_index": int(row_index),
        "customer_id": customer_id,
        "column": column,
        "anomaly_type": anomaly_type,
        "original_value": original_value,
        "corrupted_value": corrupted_value
    })


# ============================================================
# 3. GENERATE CLEAN SYNTHETIC BANKING DATA
# ============================================================

n = N_ROWS

customer_id = np.array(
    [f"C{i:06d}" for i in range(1, n + 1)]
)

# ---------------- Customer profile ----------------

age = np.clip(
    np.round(rng.normal(38, 10, n)),
    21, 70
).astype(int)

gender = choice(
    ["Male", "Female", "Other"],
    [0.62, 0.36, 0.02],
    n
)

marital_status = choice(
    ["Single", "Married", "Divorced", "Widowed"],
    [0.35, 0.52, 0.09, 0.04],
    n
)

education = choice(
    ["High School", "Graduate", "Postgraduate", "Other"],
    [0.20, 0.52, 0.23, 0.05],
    n
)

dependents = np.clip(
    rng.poisson(1.5, n),
    0, 5
).astype(int)

residence_type = choice(
    ["Owned", "Rented", "Family"],
    [0.46, 0.40, 0.14],
    n
)

# ---------------- Employment ----------------

employment_type = choice(
    ["Salaried", "Self-Employed", "Business", "Student", "Unemployed"],
    [0.53, 0.16, 0.15, 0.04, 0.12],
    n
)

max_employment_years = np.maximum(age - 18, 0)

employment_years = np.floor(
    rng.random(n) * np.minimum(max_employment_years, 35)
).astype(int)

student_mask = employment_type == "Student"
employment_years[student_mask] = rng.integers(
    0, 3, size=student_mask.sum()
)

unemployed_mask = employment_type == "Unemployed"
employment_years[unemployed_mask] = 0

# ---------------- Income ----------------

education_factor = pd.Series(education).map({
    "High School": 0.75,
    "Graduate": 1.00,
    "Postgraduate": 1.30,
    "Other": 0.85
}).to_numpy()

employment_factor = pd.Series(employment_type).map({
    "Salaried": 1.00,
    "Self-Employed": 1.08,
    "Business": 1.20,
    "Student": 0.38,
    "Unemployed": 0.32
}).to_numpy()

income_base = (
    220_000
    + (age - 21) * 10_500
    + employment_years * 18_000
)

income_noise = rng.lognormal(
    mean=0,
    sigma=0.38,
    size=n
)

annual_income = (
    income_base
    * education_factor
    * employment_factor
    * income_noise
)

annual_income = np.maximum(
    annual_income,
    100_000
)

annual_income = (
    np.round(annual_income / 1000)
    * 1000
).astype(int)

monthly_income = np.round(
    annual_income / 12
).astype(int)

# ---------------- Credit history ----------------

credit_score = (
    690
    + 0.55 * (annual_income / 100_000 - 5) * 8
    + 4 * (education == "Postgraduate")
    + rng.normal(0, 55, n)
)

credit_score = np.clip(
    np.round(credit_score),
    300, 850
).astype(int)

existing_loans = np.clip(
    rng.poisson(1.35, n)
    + (annual_income > 1_500_000).astype(int),
    0, 7
).astype(int)

previous_defaults = np.clip(
    rng.poisson(0.18, n),
    0, 4
).astype(int)

late_payments_12m = np.clip(
    rng.poisson(0.65, n)
    + (previous_defaults > 0).astype(int),
    0, 8
).astype(int)

credit_utilization = np.clip(
    rng.beta(2.4, 4.2, n)
    + 0.04 * previous_defaults,
    0.02, 0.98
)

credit_utilization = np.round(
    credit_utilization, 3
)

months_since_last_delinquency = np.where(
    late_payments_12m > 0,
    np.clip(
        rng.normal(14, 10, n),
        0, 60
    ),
    60
)

months_since_last_delinquency = np.round(
    months_since_last_delinquency
).astype(int)

# ---------------- Loan data ----------------

loan_purpose = choice(
    ["Home", "Education", "Personal", "Auto", "Medical", "Business"],
    [0.23, 0.10, 0.27, 0.18, 0.07, 0.15],
    n
)

loan_term_months = choice(
    [12, 24, 36, 48, 60, 72, 84],
    [0.08, 0.13, 0.27, 0.23, 0.17, 0.08, 0.04],
    n
)

loan_multiplier = pd.Series(loan_purpose).map({
    "Home": 2.6,
    "Education": 1.6,
    "Personal": 1.0,
    "Auto": 1.25,
    "Medical": 0.8,
    "Business": 1.8
}).to_numpy()

loan_amount = (
    annual_income
    * rng.uniform(0.25, 1.10, n)
    * loan_multiplier
)

loan_amount = np.clip(
    loan_amount,
    50_000,
    5_000_000
)

loan_amount = (
    np.round(loan_amount / 5000)
    * 5000
).astype(int)

interest_rate = (
    12.5
    + (700 - credit_score) * 0.018
    + previous_defaults * 0.65
    + rng.normal(0, 1.35, n)
)

interest_rate = np.clip(
    interest_rate,
    6.0,
    28.0
)

interest_rate = np.round(
    interest_rate, 2
)

loan_burden = (
    loan_amount
    / np.maximum(annual_income, 100_000)
)

dti_ratio = (
    0.15
    + 0.30 * np.clip(loan_burden, 0, 2)
    + 0.025 * existing_loans
    + 0.06 * credit_utilization
    + rng.normal(0, 0.06, n)
)

dti_ratio = np.clip(
    dti_ratio,
    0.05,
    0.90
)

dti_ratio = np.round(
    dti_ratio, 3
)

collateral_required = np.isin(
    loan_purpose,
    ["Home", "Auto", "Business"]
)

collateral_value = np.where(
    collateral_required,
    loan_amount * rng.uniform(0.8, 1.8, n),
    0
)

collateral_value = (
    np.round(collateral_value / 5000)
    * 5000
).astype(int)

# ---------------- Application dates ----------------

start_date = pd.Timestamp("2024-01-01")

application_days = rng.integers(
    0, 1000, n
)

application_date = (
    start_date
    + pd.to_timedelta(application_days, unit="D")
)

# ============================================================
# 4. CREATE DEFAULT LABEL
# ============================================================
# Default is generated from several risk factors rather than
# one simple if-condition.

risk_score = (
    -2.55
    + 0.0075 * (650 - credit_score)
    + 1.45 * np.maximum(dti_ratio - 0.35, 0)
    + 0.55 * previous_defaults
    + 0.18 * late_payments_12m
    + 0.85 * np.maximum(credit_utilization - 0.55, 0)
    + 0.22 * existing_loans
    + 0.28 * np.maximum(loan_burden - 1.0, 0)
    - 0.00000018 * np.maximum(annual_income - 600_000, 0)
    - 0.08 * np.minimum(employment_years, 10)
)

default_probability = sigmoid(risk_score)

loan_default = (
    rng.random(n) < default_probability
).astype(int)

# Make the synthetic class distribution approximately 18%.
# This is a project design choice, not a real banking statistic.
target_default_rate = 0.18

if not 0.12 <= loan_default.mean() <= 0.25:
    ranked_indices = np.argsort(risk_score)
    default_count = int(n * target_default_rate)

    loan_default[:] = 0
    loan_default[
        ranked_indices[-default_count:]
    ] = 1

# ============================================================
# 5. CREATE CLEAN DATAFRAME
# ============================================================

clean_df = pd.DataFrame({
    "customer_id": customer_id,
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
    "application_date": application_date.strftime("%Y-%m-%d"),
    "loan_default": loan_default
})

clean_df.to_csv(
    OUTPUT_DIR / "bank_loans_clean.csv",
    index=False
)

# ============================================================
# 6. INJECT ANOMALIES INTO A COPY
# ============================================================

raw_df = clean_df.copy()
anomaly_log = []

def customer_at(row_index):
    return str(raw_df.iloc[row_index]["customer_id"])


# ------------------------------------------------------------
# 6.1 Missing values
# ------------------------------------------------------------

missing_columns = [
    "age", "gender", "marital_status", "education",
    "dependents", "residence_type", "employment_type",
    "employment_years", "annual_income", "monthly_income",
    "credit_score", "existing_loans", "previous_defaults",
    "late_payments_12m", "credit_utilization",
    "months_since_last_delinquency", "loan_amount",
    "loan_term_months", "interest_rate", "dti_ratio",
    "collateral_value", "loan_purpose"
]

missing_operations = int(
    N_ROWS * 0.06
)

for _ in range(missing_operations):
    row = int(rng.integers(0, len(raw_df)))
    col = str(rng.choice(missing_columns))

    old_value = raw_df.at[row, col]

    if pd.isna(old_value):
        continue

    raw_df.at[row, col] = np.nan

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        col,
        "missing_value",
        old_value,
        np.nan
    )


# ------------------------------------------------------------
# 6.2 Exact duplicate rows
# ------------------------------------------------------------

duplicate_count = int(
    N_ROWS * 0.015
)

source_rows = rng.choice(
    N_ROWS,
    size=duplicate_count,
    replace=False
)

duplicate_rows = [
    raw_df.iloc[i].copy()
    for i in source_rows
]

duplicate_df = pd.DataFrame(
    duplicate_rows
)

raw_df = pd.concat(
    [raw_df, duplicate_df],
    ignore_index=True
)

original_length = N_ROWS

for offset, source_row in enumerate(source_rows):
    new_row = original_length + offset

    anomaly_log.append({
        "row_index": int(new_row),
        "customer_id": raw_df.iloc[new_row]["customer_id"],
        "column": "__row__",
        "anomaly_type": "duplicate_row",
        "original_value": f"copied_from_row_{int(source_row)}",
        "corrupted_value": f"duplicate_of_row_{int(source_row)}"
    })


# ------------------------------------------------------------
# 6.3 Numerical outliers
# ------------------------------------------------------------

outlier_generators = {
    "age": lambda: int(rng.choice([5, 95, 120, 250])),
    "annual_income": lambda: int(
        rng.choice([10_000, 50_000_000, 250_000_000])
    ),
    "monthly_income": lambda: int(
        rng.choice([2_000, 5_000_000, 20_000_000])
    ),
    "credit_score": lambda: int(
        rng.choice([80, 150, 920, 1100, 1500])
    ),
    "employment_years": lambda: int(
        rng.choice([-2, 45, 60])
    ),
    "loan_amount": lambda: int(
        rng.choice([10_000, 20_000_000, 80_000_000])
    ),
    "interest_rate": lambda: float(
        rng.choice([-5.0, 0.1, 45.0, 80.0])
    ),
    "dti_ratio": lambda: float(
        rng.choice([-0.20, 1.20, 2.50, 4.80])
    ),
    "collateral_value": lambda: int(
        rng.choice([-500_000, 50_000_000, 200_000_000])
    ),
    "existing_loans": lambda: int(
        rng.choice([-3, 15, 25])
    ),
    "previous_defaults": lambda: int(
        rng.choice([-1, 10, 20])
    )
}

outlier_operations = int(
    N_ROWS * 0.025
)

for _ in range(outlier_operations):
    row = int(rng.integers(0, len(raw_df)))
    col = str(rng.choice(list(outlier_generators.keys())))

    old_value = raw_df.at[row, col]

    if pd.isna(old_value):
        continue

    new_value = outlier_generators[col]()

    raw_df.at[row, col] = new_value

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        col,
        "numeric_outlier",
        old_value,
        new_value
    )


# ------------------------------------------------------------
# 6.4 Invalid categorical values
# ------------------------------------------------------------

categorical_errors = {
    "gender": [
        "Malee", "Femal", "M", "F", "UNKNOWN"
    ],
    "marital_status": [
        "Maried", "single", "DIVORCEDDDD", "N/A"
    ],
    "education": [
        "Graduatee", "GRAD", "Bachelorss", "Unknown123"
    ],
    "residence_type": [
        "Ownedd", "Rent", "Parent_Home", "UNKNOWN"
    ],
    "employment_type": [
        "Salariedd", "Self employed",
        "BUSSINESS", "Studnt", "N/A"
    ],
    "loan_purpose": [
        "Homee", "Persnal", "AUTO1",
        "Busines", "UnknownLoan"
    ]
}

category_operations = int(
    N_ROWS * 0.015
)

for _ in range(category_operations):
    row = int(rng.integers(0, len(raw_df)))
    col = str(
        rng.choice(list(categorical_errors.keys()))
    )

    old_value = raw_df.at[row, col]

    if pd.isna(old_value):
        continue

    new_value = str(
        rng.choice(categorical_errors[col])
    )

    raw_df.at[row, col] = new_value

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        col,
        "invalid_category",
        old_value,
        new_value
    )


# ------------------------------------------------------------
# 6.5 Negative values
# ------------------------------------------------------------

negative_columns = [
    "annual_income",
    "monthly_income",
    "loan_amount",
    "collateral_value",
    "interest_rate",
    "dti_ratio"
]

negative_operations = int(
    N_ROWS * 0.008
)

for _ in range(negative_operations):
    row = int(rng.integers(0, len(raw_df)))
    col = str(rng.choice(negative_columns))

    old_value = raw_df.at[row, col]

    if pd.isna(old_value):
        continue

    new_value = -abs(float(old_value))

    raw_df.at[row, col] = new_value

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        col,
        "negative_value",
        old_value,
        new_value
    )


# ------------------------------------------------------------
# 6.6 Logical inconsistencies
# ------------------------------------------------------------

logical_operations = int(
    N_ROWS * 0.012
)

for _ in range(logical_operations):

    row = int(
        rng.integers(0, len(raw_df))
    )

    pattern = str(
        rng.choice([
            "employment_gt_age",
            "student_high_employment",
            "huge_loan",
            "invalid_ratio",
            "invalid_dependents"
        ])
    )

    if pattern == "employment_gt_age":

        old_value = raw_df.at[row, "employment_years"]

        if pd.isna(old_value):
            continue

        age_value = raw_df.at[row, "age"]

        if pd.isna(age_value):
            continue

        new_value = int(
            float(age_value) + 20
        )

        raw_df.at[row, "employment_years"] = new_value

        add_log(
            anomaly_log,
            row,
            customer_at(row),
            "employment_years",
            "logical_inconsistency",
            old_value,
            new_value
        )

    elif pattern == "student_high_employment":

        old_type = raw_df.at[row, "employment_type"]

        raw_df.at[row, "employment_type"] = "Student"

        add_log(
            anomaly_log,
            row,
            customer_at(row),
            "employment_type",
            "logical_inconsistency",
            old_type,
            "Student"
        )

        old_years = raw_df.at[row, "employment_years"]

        if not pd.isna(old_years):

            new_years = int(
                rng.integers(15, 30)
            )

            raw_df.at[row, "employment_years"] = new_years

            add_log(
                anomaly_log,
                row,
                customer_at(row),
                "employment_years",
                "logical_inconsistency",
                old_years,
                new_years
            )

    elif pattern == "huge_loan":

        old_value = raw_df.at[row, "loan_amount"]

        income_value = raw_df.at[row, "annual_income"]

        if pd.isna(old_value) or pd.isna(income_value):
            continue

        new_value = int(
            max(
                10_000_000,
                float(income_value) * 25
            )
        )

        raw_df.at[row, "loan_amount"] = new_value

        add_log(
            anomaly_log,
            row,
            customer_at(row),
            "loan_amount",
            "logical_inconsistency",
            old_value,
            new_value
        )

    elif pattern == "invalid_ratio":

        col = str(
            rng.choice([
                "credit_utilization",
                "dti_ratio"
            ])
        )

        old_value = raw_df.at[row, col]

        if pd.isna(old_value):
            continue

        new_value = float(
            rng.choice([1.2, 1.5, 2.2])
        )

        raw_df.at[row, col] = new_value

        add_log(
            anomaly_log,
            row,
            customer_at(row),
            col,
            "logical_inconsistency",
            old_value,
            new_value
        )

    else:

        old_value = raw_df.at[row, "dependents"]

        if pd.isna(old_value):
            continue

        new_value = int(
            rng.choice([-1, 15, 25])
        )

        raw_df.at[row, "dependents"] = new_value

        add_log(
            anomaly_log,
            row,
            customer_at(row),
            "dependents",
            "logical_inconsistency",
            old_value,
            new_value
        )


# ------------------------------------------------------------
# 6.7 Date errors
# ------------------------------------------------------------

date_operations = int(
    N_ROWS * 0.006
)

bad_dates = [
    "2035-12-45",
    "1995-02-30",
    "2050-01-01",
    "not-a-date",
    "2026/99/99"
]

for _ in range(date_operations):

    row = int(
        rng.integers(0, len(raw_df))
    )

    old_value = raw_df.at[
        row,
        "application_date"
    ]

    if pd.isna(old_value):
        continue

    new_value = str(
        rng.choice(bad_dates)
    )

    raw_df.at[
        row,
        "application_date"
    ] = new_value

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        "application_date",
        "date_error",
        old_value,
        new_value
    )


# ------------------------------------------------------------
# 6.8 Formatting inconsistencies
# ------------------------------------------------------------

format_columns = [
    "gender",
    "marital_status",
    "education",
    "residence_type",
    "employment_type",
    "loan_purpose"
]

format_operations = int(
    N_ROWS * 0.012
)

for _ in range(format_operations):

    row = int(
        rng.integers(0, len(raw_df))
    )

    col = str(
        rng.choice(format_columns)
    )

    old_value = raw_df.at[row, col]

    if pd.isna(old_value):
        continue

    mode = str(
        rng.choice([
            "lowercase",
            "uppercase",
            "spaces"
        ])
    )

    if mode == "lowercase":
        new_value = str(old_value).lower()

    elif mode == "uppercase":
        new_value = str(old_value).upper()

    else:
        new_value = f"  {old_value}  "

    if new_value == old_value:
        continue

    raw_df.at[row, col] = new_value

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        col,
        "formatting_inconsistency",
        old_value,
        new_value
    )


# ------------------------------------------------------------
# 6.9 Near-duplicate changes
# ------------------------------------------------------------

near_duplicate_operations = int(
    N_ROWS * 0.006
)

for _ in range(near_duplicate_operations):

    row = int(
        rng.integers(0, len(raw_df))
    )

    col = str(
        rng.choice([
            "annual_income",
            "monthly_income",
            "loan_amount",
            "interest_rate",
            "credit_score"
        ])
    )

    old_value = raw_df.at[row, col]

    if pd.isna(old_value):
        continue

    if col in [
        "annual_income",
        "monthly_income",
        "loan_amount"
    ]:

        delta = int(
            max(
                1000,
                abs(float(old_value))
                * rng.uniform(0.005, 0.02)
            )
        )

        new_value = int(
            old_value + delta
        )

    elif col == "interest_rate":

        new_value = round(
            float(old_value)
            + float(rng.uniform(0.1, 0.4)),
            2
        )

    else:

        new_value = int(
            old_value
            + rng.choice([-2, -1, 1, 2])
        )

    raw_df.at[row, col] = new_value

    add_log(
        anomaly_log,
        row,
        customer_at(row),
        col,
        "near_duplicate_change",
        old_value,
        new_value
    )


# ============================================================
# 7. SAVE RAW DATA + ANOMALY LOG
# ============================================================

raw_df.to_csv(
    OUTPUT_DIR / "bank_loans_raw.csv",
    index=False
)

anomaly_log_df = pd.DataFrame(
    anomaly_log
)

anomaly_log_df.to_csv(
    OUTPUT_DIR / "anomaly_log.csv",
    index=False
)


# ============================================================
# 8. DATA DICTIONARY
# ============================================================

dictionary = pd.DataFrame([
    ["customer_id", "Unique synthetic customer identifier", "string", "C000001 etc."],
    ["age", "Customer age", "integer", "21-70 in clean data"],
    ["gender", "Customer gender", "category", "Male/Female/Other"],
    ["marital_status", "Marital status", "category", "Single/Married/Divorced/Widowed"],
    ["education", "Education level", "category", "High School/Graduate/Postgraduate/Other"],
    ["dependents", "Number of dependents", "integer", "0-5"],
    ["residence_type", "Residence type", "category", "Owned/Rented/Family"],
    ["employment_type", "Employment category", "category", "Salaried/Self-Employed/Business/Student/Unemployed"],
    ["employment_years", "Employment duration", "integer", ">=0"],
    ["annual_income", "Annual income in INR", "numeric", ">0"],
    ["monthly_income", "Monthly income in INR", "numeric", ">0"],
    ["credit_score", "Synthetic credit score", "integer", "300-850"],
    ["existing_loans", "Existing loan count", "integer", ">=0"],
    ["previous_defaults", "Previous default count", "integer", ">=0"],
    ["late_payments_12m", "Late payments in previous 12 months", "integer", ">=0"],
    ["credit_utilization", "Credit utilization ratio", "float", "0-1"],
    ["months_since_last_delinquency", "Months since last delinquency", "integer", "0-60"],
    ["loan_amount", "Requested loan amount", "numeric", ">0"],
    ["loan_term_months", "Loan term", "integer", "12-84"],
    ["interest_rate", "Annual interest rate", "float", "6-28"],
    ["dti_ratio", "Debt-to-income ratio", "float", "0-1"],
    ["collateral_value", "Collateral value", "numeric", ">=0"],
    ["loan_purpose", "Loan purpose", "category", "Home/Education/Personal/Auto/Medical/Business"],
    ["application_date", "Application date", "date", "YYYY-MM-DD"],
    ["loan_default", "Target variable", "binary", "0=Non-default, 1=Default"]
], columns=[
    "column_name",
    "description",
    "data_type",
    "valid_range_or_values"
])

dictionary.to_csv(
    OUTPUT_DIR / "data_dictionary.csv",
    index=False
)


# ============================================================
# 9. SUMMARY
# ============================================================

summary_text = f"""
BANK LOAN DATASET GENERATION SUMMARY
=====================================

Random seed: {SEED}

Clean rows:
{len(clean_df):,}

Raw rows:
{len(raw_df):,}

Number of columns:
{len(clean_df.columns)}

Clean default rate:
{clean_df["loan_default"].mean() * 100:.2f}%

Exact duplicate rows intentionally added:
{duplicate_count:,}

Anomaly events logged:
{len(anomaly_log_df):,}

Generated files:
- bank_loans_clean.csv
- bank_loans_raw.csv
- anomaly_log.csv
- data_dictionary.csv
"""

(
    OUTPUT_DIR / "dataset_summary.txt"
).write_text(
    summary_text,
    encoding="utf-8"
)


# ============================================================
# 10. CONSOLE OUTPUT
# ============================================================

print("=" * 60)
print("BANK LOAN DATASET GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Clean dataset : {OUTPUT_DIR / 'bank_loans_clean.csv'}")
print(f"Raw dataset   : {OUTPUT_DIR / 'bank_loans_raw.csv'}")
print(f"Anomaly log   : {OUTPUT_DIR / 'anomaly_log.csv'}")
print(f"Data dictionary: {OUTPUT_DIR / 'data_dictionary.csv'}")
print()
print(f"Clean rows    : {len(clean_df):,}")
print(f"Raw rows      : {len(raw_df):,}")
print(f"Columns       : {len(clean_df.columns)}")
print(
    f"Default rate  : "
    f"{clean_df['loan_default'].mean() * 100:.2f}%"
)
print(
    f"Anomaly events: "
    f"{len(anomaly_log_df):,}"
)
print("=" * 60)
