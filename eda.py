import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

INPUT_FILE = "bank_loan_dataset_output/bank_loans_cleaned.csv"
EDA_DIR = "bank_loan_dataset_output/eda"

os.makedirs(EDA_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("BANK LOAN DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 2. BASIC INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(df.duplicated().sum())


# ------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

descriptive_stats = numeric_df.describe().T

descriptive_stats.to_csv(
    os.path.join(EDA_DIR, "descriptive_statistics.csv")
)

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(descriptive_stats)


# ------------------------------------------------------------
# 4. TARGET VARIABLE ANALYSIS
# ------------------------------------------------------------

target_counts = df["loan_default"].value_counts().sort_index()

target_distribution = pd.DataFrame({
    "loan_default": target_counts.index,
    "count": target_counts.values
})

target_distribution["percentage"] = (
    target_distribution["count"] /
    target_distribution["count"].sum() * 100
)

target_distribution.to_csv(
    os.path.join(EDA_DIR, "default_distribution.csv"),
    index=False
)

print("\n" + "=" * 60)
print("LOAN DEFAULT DISTRIBUTION")
print("=" * 60)

print(target_distribution)


# ------------------------------------------------------------
# 5. DEFAULT DISTRIBUTION CHART
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

labels = ["Non-Default", "Default"]

plt.bar(
    labels,
    target_distribution["count"]
)

plt.title("Loan Default Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "01_default_distribution.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 6. AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=20,
    edgecolor="black"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "02_age_distribution.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 7. ANNUAL INCOME DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["annual_income"],
    bins=30,
    edgecolor="black"
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "03_income_distribution.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 8. CREDIT SCORE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["credit_score"],
    bins=25,
    edgecolor="black"
)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "04_credit_score_distribution.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 9. LOAN AMOUNT DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["loan_amount"],
    bins=30,
    edgecolor="black"
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "05_loan_amount_distribution.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 10. DEFAULT RATE BY GENDER
# ------------------------------------------------------------

gender_default = (
    df.groupby("gender")["loan_default"]
    .agg(["count", "mean"])
    .reset_index()
)

gender_default["default_rate_percent"] = (
    gender_default["mean"] * 100
)

gender_default.drop(columns=["mean"], inplace=True)

gender_default.to_csv(
    os.path.join(EDA_DIR, "default_by_gender.csv"),
    index=False
)

plt.figure(figsize=(7, 5))

plt.bar(
    gender_default["gender"],
    gender_default["default_rate_percent"]
)

plt.title("Default Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Default Rate (%)")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "06_default_by_gender.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 11. DEFAULT RATE BY EMPLOYMENT TYPE
# ------------------------------------------------------------

employment_default = (
    df.groupby("employment_type")["loan_default"]
    .agg(["count", "mean"])
    .reset_index()
)

employment_default["default_rate_percent"] = (
    employment_default["mean"] * 100
)

employment_default.drop(columns=["mean"], inplace=True)

employment_default.to_csv(
    os.path.join(EDA_DIR, "default_by_employment.csv"),
    index=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    employment_default["employment_type"],
    employment_default["default_rate_percent"]
)

plt.title("Default Rate by Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "07_default_by_employment.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 12. DEFAULT RATE BY LOAN PURPOSE
# ------------------------------------------------------------

purpose_default = (
    df.groupby("loan_purpose")["loan_default"]
    .agg(["count", "mean"])
    .reset_index()
)

purpose_default["default_rate_percent"] = (
    purpose_default["mean"] * 100
)

purpose_default.drop(columns=["mean"], inplace=True)

purpose_default.to_csv(
    os.path.join(EDA_DIR, "default_by_loan_purpose.csv"),
    index=False
)

plt.figure(figsize=(10, 5))

plt.bar(
    purpose_default["loan_purpose"],
    purpose_default["default_rate_percent"]
)

plt.title("Default Rate by Loan Purpose")
plt.xlabel("Loan Purpose")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "08_default_by_loan_purpose.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 13. CREDIT SCORE BANDS
# ------------------------------------------------------------

bins = [300, 550, 650, 750, 850]

labels = [
    "300-549",
    "550-649",
    "650-749",
    "750-849"
]

df["credit_score_band"] = pd.cut(
    df["credit_score"],
    bins=bins,
    labels=labels,
    right=False
)

credit_band_default = (
    df.groupby(
        "credit_score_band",
        observed=False
    )["loan_default"]
    .agg(["count", "mean"])
    .reset_index()
)

credit_band_default["default_rate_percent"] = (
    credit_band_default["mean"] * 100
)

credit_band_default.drop(columns=["mean"], inplace=True)

credit_band_default.to_csv(
    os.path.join(EDA_DIR, "default_by_credit_score_band.csv"),
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    credit_band_default["credit_score_band"].astype(str),
    credit_band_default["default_rate_percent"]
)

plt.title("Default Rate by Credit Score Band")
plt.xlabel("Credit Score Band")
plt.ylabel("Default Rate (%)")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "09_default_by_credit_score.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 14. DEFAULT RATE BY EDUCATION
# ------------------------------------------------------------

education_default = (
    df.groupby("education")["loan_default"]
    .agg(["count", "mean"])
    .reset_index()
)

education_default["default_rate_percent"] = (
    education_default["mean"] * 100
)

education_default.drop(columns=["mean"], inplace=True)

education_default.to_csv(
    os.path.join(EDA_DIR, "default_by_education.csv"),
    index=False
)


# ------------------------------------------------------------
# 15. DEFAULT RATE BY MARITAL STATUS
# ------------------------------------------------------------

marital_default = (
    df.groupby("marital_status")["loan_default"]
    .agg(["count", "mean"])
    .reset_index()
)

marital_default["default_rate_percent"] = (
    marital_default["mean"] * 100
)

marital_default.drop(columns=["mean"], inplace=True)

marital_default.to_csv(
    os.path.join(EDA_DIR, "default_by_marital_status.csv"),
    index=False
)


# ------------------------------------------------------------
# 16. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = numeric_df.corr()

correlation.to_csv(
    os.path.join(EDA_DIR, "correlation_matrix.csv")
)

# Heatmap using matplotlib only
plt.figure(figsize=(14, 11))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(EDA_DIR, "10_correlation_heatmap.png"),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 17. CORRELATION WITH TARGET
# ------------------------------------------------------------

target_correlation = (
    correlation["loan_default"]
    .sort_values(ascending=False)
    .to_frame("correlation_with_default")
)

target_correlation.to_csv(
    os.path.join(EDA_DIR, "correlation_with_default.csv")
)

print("\n" + "=" * 60)
print("CORRELATION WITH LOAN DEFAULT")
print("=" * 60)

print(target_correlation)


# ------------------------------------------------------------
# 18. SUMMARY REPORT
# ------------------------------------------------------------

summary_file = os.path.join(
    EDA_DIR,
    "eda_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("BANK LOAN DATASET - EDA SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Rows: {df.shape[0]}\n")
    f.write(f"Columns: {df.shape[1]}\n\n")

    f.write(
        f"Default Rate: "
        f"{df['loan_default'].mean() * 100:.2f}%\n"
    )

    f.write(
        f"Non-Default Rate: "
        f"{(1 - df['loan_default'].mean()) * 100:.2f}%\n\n"
    )

    f.write("Missing Values: ")
    f.write(str(df.isnull().sum().sum()))
    f.write("\n")

    f.write("Duplicate Rows: ")
    f.write(str(df.duplicated().sum()))
    f.write("\n\n")

    f.write("Correlation with Loan Default:\n")
    f.write(target_correlation.to_string())


# ------------------------------------------------------------
# 19. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nEDA files saved in:")
print(EDA_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(EDA_DIR)):
    print(" -", file)

print("\nNext Phase: Machine Learning Prediction")