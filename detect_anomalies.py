import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = Path(
    "bank_loan_dataset_output/bank_loans_raw.csv"
)

df = pd.read_csv(FILE_PATH)

print("=" * 70)
print("BANK LOAN DATASET - ANOMALY DETECTION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. MISSING VALUE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("1. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = (
    missing_count / len(df) * 100
).round(2)

missing_report = pd.DataFrame({
    "column": missing_count.index,
    "missing_count": missing_count.values,
    "missing_percentage": missing_percentage.values
})

missing_report = missing_report[
    missing_report["missing_count"] > 0
].sort_values(
    "missing_count",
    ascending=False
)

print(missing_report.to_string(index=False))


# ============================================================
# 3. DUPLICATE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("2. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(
    f"Exact duplicate rows: {duplicate_count}"
)

duplicate_customer_ids = (
    df["customer_id"].duplicated().sum()
)

print(
    f"Duplicate customer IDs: "
    f"{duplicate_customer_ids}"
)


# ============================================================
# 4. NUMERICAL OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 70)
print("3. NUMERICAL OUTLIER ANALYSIS")
print("=" * 70)

numeric_columns = [
    "age",
    "dependents",
    "employment_years",
    "annual_income",
    "monthly_income",
    "credit_score",
    "existing_loans",
    "previous_defaults",
    "late_payments_12m",
    "credit_utilization",
    "months_since_last_delinquency",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "dti_ratio",
    "collateral_value"
]

outlier_results = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    count = outlier_mask.sum()

    outlier_results.append({
        "column": column,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": int(count)
    })

outlier_report = pd.DataFrame(
    outlier_results
)

print(
    outlier_report[
        [
            "column",
            "lower_bound",
            "upper_bound",
            "outlier_count"
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. INVALID CATEGORICAL VALUE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("4. INVALID CATEGORY ANALYSIS")
print("=" * 70)

valid_categories = {

    "gender": [
        "Male",
        "Female",
        "Other"
    ],

    "marital_status": [
        "Single",
        "Married",
        "Divorced",
        "Widowed"
    ],

    "education": [
        "High School",
        "Graduate",
        "Postgraduate",
        "Other"
    ],

    "residence_type": [
        "Owned",
        "Rented",
        "Family"
    ],

    "employment_type": [
        "Salaried",
        "Self-Employed",
        "Business",
        "Student",
        "Unemployed"
    ],

    "loan_purpose": [
        "Home",
        "Education",
        "Personal",
        "Auto",
        "Medical",
        "Business"
    ]
}

invalid_category_results = []

for column, valid_values in valid_categories.items():

    invalid_mask = ~df[column].isin(
        valid_values
    )

    invalid_count = invalid_mask.sum()

    invalid_values = (
        df.loc[
            invalid_mask,
            column
        ]
        .dropna()
        .unique()
        .tolist()
    )

    invalid_category_results.append({
        "column": column,
        "invalid_count": int(invalid_count),
        "invalid_values": str(
            invalid_values
        )
    })

invalid_category_report = pd.DataFrame(
    invalid_category_results
)

print(
    invalid_category_report.to_string(
        index=False
    )
)


# ============================================================
# 6. NEGATIVE VALUE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("5. NEGATIVE VALUE ANALYSIS")
print("=" * 70)

non_negative_columns = [
    "age",
    "dependents",
    "employment_years",
    "annual_income",
    "monthly_income",
    "credit_score",
    "existing_loans",
    "previous_defaults",
    "late_payments_12m",
    "credit_utilization",
    "months_since_last_delinquency",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "dti_ratio",
    "collateral_value"
]

negative_results = []

for column in non_negative_columns:

    negative_mask = (
        df[column] < 0
    )

    negative_count = negative_mask.sum()

    negative_results.append({
        "column": column,
        "negative_count": int(
            negative_count
        )
    })

negative_report = pd.DataFrame(
    negative_results
)

print(
    negative_report[
        negative_report[
            "negative_count"
        ] > 0
    ].to_string(index=False)
)


# ============================================================
# 7. LOGICAL INCONSISTENCY DETECTION
# ============================================================

print("\n" + "=" * 70)
print("6. LOGICAL INCONSISTENCY ANALYSIS")
print("=" * 70)


# Rule 1:
# Employment years should not exceed age - 18.

employment_error = (
    df["employment_years"]
    > (df["age"] - 18)
)

print(
    "Employment years > age - 18:",
    employment_error.sum()
)


# Rule 2:
# Credit utilization should be between 0 and 1.

credit_utilization_error = (
    (df["credit_utilization"] < 0) |
    (df["credit_utilization"] > 1)
)

print(
    "Invalid credit utilization:",
    credit_utilization_error.sum()
)


# Rule 3:
# DTI should be between 0 and 1.

dti_error = (
    (df["dti_ratio"] < 0) |
    (df["dti_ratio"] > 1)
)

print(
    "Invalid DTI ratio:",
    dti_error.sum()
)


# Rule 4:
# Credit score should be 300-850.

credit_score_error = (
    (df["credit_score"] < 300) |
    (df["credit_score"] > 850)
)

print(
    "Invalid credit score:",
    credit_score_error.sum()
)


# Rule 5:
# Age should be 21-70.

age_error = (
    (df["age"] < 21) |
    (df["age"] > 70)
)

print(
    "Invalid age:",
    age_error.sum()
)


# ============================================================
# 8. DATE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("7. DATE VALIDATION")
print("=" * 70)

parsed_dates = pd.to_datetime(
    df["application_date"],
    errors="coerce"
)

invalid_dates = parsed_dates.isna()

print(
    "Invalid dates:",
    invalid_dates.sum()
)


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("8. TARGET VARIABLE ANALYSIS")
print("=" * 70)

print(
    df["loan_default"]
    .value_counts()
)

print("\nPercentage:")

print(
    (
        df["loan_default"]
        .value_counts(
            normalize=True
        ) * 100
    ).round(2)
)


# ============================================================
# 10. SAVE REPORTS
# ============================================================

OUTPUT_DIR = Path(
    "bank_loan_dataset_output"
)

missing_report.to_csv(
    OUTPUT_DIR / "missing_value_report.csv",
    index=False
)

outlier_report.to_csv(
    OUTPUT_DIR / "outlier_report.csv",
    index=False
)

invalid_category_report.to_csv(
    OUTPUT_DIR / "invalid_category_report.csv",
    index=False
)

negative_report.to_csv(
    OUTPUT_DIR / "negative_value_report.csv",
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY DETECTION COMPLETED")
print("=" * 70)

print(
    f"Rows analysed: {len(df):,}"
)

print(
    f"Columns analysed: {len(df.columns)}"
)

print(
    f"Missing cells: "
    f"{df.isnull().sum().sum():,}"
)

print(
    f"Exact duplicate rows: "
    f"{df.duplicated().sum():,}"
)

print("\nReports saved inside:")

print(
    "bank_loan_dataset_output/"
)

print(
    "- missing_value_report.csv"
)

print(
    "- outlier_report.csv"
)

print(
    "- invalid_category_report.csv"
)

print(
    "- negative_value_report.csv"
)