import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# ============================================================
# PHASE 4: MACHINE LEARNING
# BANK LOAN DEFAULT PREDICTION
# ============================================================

INPUT_FILE = "bank_loan_dataset_output/bank_loans_cleaned.csv"
ML_DIR = "bank_loan_dataset_output/ml"

os.makedirs(ML_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("BANK LOAN DEFAULT PREDICTION - MACHINE LEARNING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(df["loan_default"].value_counts())


# ============================================================
# 2. BASIC PREPARATION
# ============================================================

# Convert application date into useful numerical features
df["application_date"] = pd.to_datetime(
    df["application_date"],
    errors="coerce"
)

df["application_year"] = df["application_date"].dt.year
df["application_month"] = df["application_date"].dt.month
df["application_day"] = df["application_date"].dt.day

# Remove original date because ML models cannot directly use
# a date string
df.drop(columns=["application_date"], inplace=True)

# Remove customer ID because it is only an identifier
df.drop(columns=["customer_id"], inplace=True)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["loan_default"])
y = df["loan_default"]

print("\nNumber of Features:", X.shape[1])
print("Target:", "loan_default")


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING PIPELINE
# ============================================================

# Numerical preprocessing
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Store results
    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    trained_models[model_name] = pipeline

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Default", "Default"]
    )

    disp.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(
            ML_DIR,
            filename
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

results_df.to_csv(
    os.path.join(
        ML_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# 10. MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.18

plt.bar(
    x - 1.5 * width,
    results_df["accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    results_df["precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    results_df["recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    results_df["f1_score"],
    width,
    label="F1 Score"
)

plt.xticks(
    x,
    results_df["model"]
)

plt.ylabel("Score")

plt.xlabel("Model")

plt.title("Machine Learning Model Comparison")

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        ML_DIR,
        "model_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 11. SELECT MODEL FOR SAVING
# ============================================================

# For this educational project we use ROC-AUC as the
# selection metric. The choice should be interpreted in
# the context of the project objective.

best_model_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("SELECTED MODEL FOR DEPLOYMENT")
print("=" * 70)

print("Model:", best_model_name)

# Save model
model_path = os.path.join(
    ML_DIR,
    "loan_default_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("Saved to:", model_path)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

model_object = best_model.named_steps["model"]
fitted_preprocessor = best_model.named_steps["preprocessor"]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance = None

if hasattr(model_object, "feature_importances_"):

    importance = model_object.feature_importances_

elif hasattr(model_object, "coef_"):

    importance = np.abs(
        model_object.coef_[0]
    )


if importance is not None:

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False
        )
    )

    feature_importance.to_csv(
        os.path.join(
            ML_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    print("\nTop 15 Features:")

    print(
        feature_importance
        .head(15)
        .to_string(index=False)
    )

    # Plot top 15
    top_features = (
        feature_importance
        .head(15)
        .sort_values(
            by="importance"
        )
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        f"Top Features - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ML_DIR,
            "feature_importance.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = X_test.copy()

test_predictions["actual_default"] = y_test.values

test_predictions["predicted_default"] = (
    best_model.predict(X_test)
)

test_predictions["default_probability"] = (
    best_model.predict_proba(X_test)[:, 1]
)

test_predictions.to_csv(
    os.path.join(
        ML_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 14. FINAL REPORT
# ============================================================

report_path = os.path.join(
    ML_DIR,
    "ml_summary.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BANK LOAN DEFAULT PREDICTION - ML SUMMARY\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Dataset size: {df.shape}\n"
    )

    f.write(
        f"Training samples: {len(X_train)}\n"
    )

    f.write(
        f"Testing samples: {len(X_test)}\n\n"
    )

    f.write(
        "MODEL COMPARISON\n"
    )

    f.write("-" * 60 + "\n")

    f.write(
        results_df.to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        f"Selected model: {best_model_name}\n"
    )

    f.write(
        "Selection metric: ROC-AUC\n"
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING PHASE COMPLETED")
print("=" * 70)

print("\nML files saved in:")
print(ML_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(ML_DIR)):
    print(" -", file)