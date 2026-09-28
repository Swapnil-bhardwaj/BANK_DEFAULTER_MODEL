import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "bank_loan_dataset_output/bank_loans_raw.csv"
)

OUTPUT_DIR = Path(
    "bank_loan_dataset_output"
)

OUTPUT_FILE = OUTPUT_DIR / "bank_loans_cleaned.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

print("=" * 70)
print("BANK LOAN DATA CLEANING")
print("=" * 70)

print(f"\nOriginal shape: {df.shape}")


# ============================================================
# 1. REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    keep="first"
)

removed_duplicates = (
    before - len(df)
)

print("\n1. Duplicate removal")
print(
    f"Exact duplicate rows removed: "
    f"{removed_duplicates}"
)


# ============================================================
# 2. STANDARDIZE CATEGORICAL VALUES
# ============================================================

print("\n2. Standardizing categorical values")

category_maps = {

    "gender": {
        "male": "Male",
        "m": "Male",
        "malee": "Male",
        "female": "Female",
        "f": "Female",
        "femal": "Female",
        "other": "Other",
        "unknown": np.nan
    },

    "marital_status": {
        "single": "Single",
        "maried": "Married",
        "married": "Married",
        "divorced": "Divorced",
        "divorcedddd": "Divorced",
        "widowed": "Widowed",
        "n/a": np.nan
    },

    "education": {
        "high school": "High School",
        "graduate": "Graduate",
        "graduatee": "Graduate",
        "grad": "Graduate",
        "postgraduate": "Postgraduate",
        "bachelorss": "Graduate",
        "other": "Other",
        "unknown123": np.nan
    },

    "residence_type": {
        "owned": "Owned",
        "ownedd": "Owned",
        "rent": "Rented",
        "rented": "Rented",
        "family": "Family",
        "parent_home": "Family",
        "unknown": np.nan
    },

    "employment_type": {
        "salaried": "Salaried",
        "salariedd": "Salaried",
        "self-employed": "Self-Employed",
        "self employed": "Self-Employed",
        "business": "Business",
        "bussiness": "Business",
        "student": "Student",
        "studnt": "Student",
        "unemployed": "Unemployed",
        "n/a": np.nan
    },

    "loan_purpose": {
        "home": "Home",
        "homee": "Home",
        "education": "Education",
        "personal": "Personal",
        "persnal": "Personal",
        "auto": "Auto",
        "auto1": "Auto",
        "medical": "Medical",
        "business": "Business",
        "busines": "Business",
        "unknownloan": np.nan
    }
}


for column, mapping in category_maps.items():

    # Remove extra spaces
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Convert using mapping
    df[column] = df[column].map(mapping)


# ============================================================
# 3. CONVERT DATE COLUMN
# ============================================================

print("\n3. Validating application dates")

df["application_date"] = pd.to_datetime(
    df["application_date"],
    errors="coerce"
)

invalid_dates = (
    df["application_date"].isna().sum()
)

print(
    f"Invalid dates converted to missing: "
    f"{invalid_dates}"
)


# ============================================================
# 4. DOMAIN VALIDATION
# ============================================================

print("\n4. Applying domain validation rules")

valid_ranges = {

    "age": (21, 70),

    "dependents": (0, 5),

    "employment_years": (0, 52),

    "annual_income": (50000, 50000000),

    "monthly_income": (4000, 10000000),

    "credit_score": (300, 850),

    "existing_loans": (0, 20),

    "previous_defaults": (0, 10),

    "late_payments_12m": (0, 20),

    "credit_utilization": (0, 1),

    "months_since_last_delinquency": (0, 120),

    "loan_amount": (50000, 20000000),

    "loan_term_months": (6, 120),

    "interest_rate": (1, 50),

    "dti_ratio": (0, 1),

    "collateral_value": (0, 100000000)
}


invalid_values_fixed = 0

for column, (lower, upper) in valid_ranges.items():

    invalid_mask = (
        (df[column] < lower) |
        (df[column] > upper)
    )

    count = invalid_mask.sum()

    if count > 0:

        df.loc[
            invalid_mask,
            column
        ] = np.nan

        invalid_values_fixed += count

        print(
            f"{column}: "
            f"{count} invalid values replaced"
            " with missing"
        )


# ============================================================
# 5. LOGICAL CONSISTENCY RULES
# ============================================================

print("\n5. Checking logical consistency")


# Employment years cannot exceed age - 18

employment_error = (
    df["employment_years"]
    > (df["age"] - 18)
)

print(
    "Employment inconsistencies:",
    employment_error.sum()
)

df.loc[
    employment_error,
    "employment_years"
] = np.nan


# Credit utilization

credit_error = (
    (df["credit_utilization"] < 0) |
    (df["credit_utilization"] > 1)
)

df.loc[
    credit_error,
    "credit_utilization"
] = np.nan


# DTI

dti_error = (
    (df["dti_ratio"] < 0) |
    (df["dti_ratio"] > 1)
)

df.loc[
    dti_error,
    "dti_ratio"
] = np.nan


# ============================================================
# 6. INCOME CONSISTENCY
# ============================================================

print("\n6. Checking income consistency")

valid_income = (
    df["annual_income"].notna() &
    df["monthly_income"].notna()
)

expected_monthly = (
    df["annual_income"] / 12
)

income_error = (
    valid_income &
    (
        df["monthly_income"]
        < expected_monthly * 0.5
    )
)

income_error |= (
    valid_income &
    (
        df["monthly_income"]
        > expected_monthly * 2
    )
)

print(
    "Income inconsistencies:",
    income_error.sum()
)

# Correct monthly income from annual income
df.loc[
    income_error,
    "monthly_income"
] = (
    df.loc[
        income_error,
        "annual_income"
    ] / 12
).round()


# ============================================================
# 7. STUDENT / UNEMPLOYED CONSISTENCY
# ============================================================

student_mask = (
    df["employment_type"]
    == "Student"
)

unemployed_mask = (
    df["employment_type"]
    == "Unemployed"
)

df.loc[
    student_mask |
    unemployed_mask,
    "employment_years"
] = 0


# ============================================================
# 8. HANDLE MISSING NUMERICAL VALUES
# ============================================================

print("\n7. Handling numerical missing values")

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

for column in numeric_columns:

    missing_before = (
        df[column].isna().sum()
    )

    if missing_before == 0:
        continue

    median_value = (
        df[column].median()
    )

    df[column] = (
        df[column]
        .fillna(median_value)
    )

    print(
        f"{column}: "
        f"{missing_before} values "
        f"filled with median "
        f"{median_value:.2f}"
    )


# ============================================================
# 9. HANDLE CATEGORICAL MISSING VALUES
# ============================================================

print("\n8. Handling categorical missing values")

categorical_columns = [
    "gender",
    "marital_status",
    "education",
    "residence_type",
    "employment_type",
    "loan_purpose"
]

for column in categorical_columns:

    missing_before = (
        df[column].isna().sum()
    )

    if missing_before == 0:
        continue

    mode_value = (
        df[column]
        .mode()[0]
    )

    df[column] = (
        df[column]
        .fillna(mode_value)
    )

    print(
        f"{column}: "
        f"{missing_before} values "
        f"filled with '{mode_value}'"
    )


# ============================================================
# 10. HANDLE DATE MISSING VALUES
# ============================================================

date_missing = (
    df["application_date"].isna().sum()
)

if date_missing > 0:

    median_date = (
        df["application_date"]
        .dropna()
        .sort_values()
        .iloc[
            len(
                df["application_date"]
                .dropna()
            ) // 2
        ]
    )

    df["application_date"] = (
        df["application_date"]
        .fillna(median_date)
    )

    print(
        f"\n{date_missing} invalid/missing "
        f"dates filled using median date:"
        f" {median_date.date()}"
    )


# ============================================================
# 11. FINAL FORMATTING
# ============================================================

df["application_date"] = (
    df["application_date"]
    .dt.strftime("%Y-%m-%d")
)


# ============================================================
# 12. FINAL DUPLICATE REMOVAL
# ============================================================

print("\n9. Final duplicate verification")

# Remove duplicates created after category standardization
before_duplicates = len(df)

df = df.drop_duplicates(
    keep="first"
)

removed_after_cleaning = (
    before_duplicates - len(df)
)

print(
    "Duplicate rows removed after cleaning:",
    removed_after_cleaning
)

# Each customer should have only one loan record
before_customer_dedup = len(df)

df = df.drop_duplicates(
    subset=["customer_id"],
    keep="first"
)

removed_customer_duplicates = (
    before_customer_dedup - len(df)
)

print(
    "Duplicate customer IDs removed:",
    removed_customer_duplicates
)


# ============================================================
# 13. FINAL QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL QUALITY CHECK")
print("=" * 70)

print(
    "Final shape:",
    df.shape
)

print(
    "Missing cells:",
    df.isna().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Duplicate customer IDs:",
    df["customer_id"].duplicated().sum()
)

print(
    "Invalid dates:",
    pd.to_datetime(
        df["application_date"],
        errors="coerce"
    ).isna().sum()
)

print(
    "Default rate:",
    round(
        df["loan_default"].mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# 14. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLEANING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nCleaned dataset saved to:\n"
    f"{OUTPUT_FILE}"
)