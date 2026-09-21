import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = "data/employee_attrition_2000_IBM_structure.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("EMPLOYEE ATTRITION MODEL DEVELOPMENT")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)


# =========================================================
# 2. REMOVE UNNECESSARY COLUMNS
# =========================================================

columns_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(
    columns=columns_to_remove,
    inplace=True,
    errors="ignore"
)


# =========================================================
# 3. TARGET ENCODING
# =========================================================

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

# Salary category
df["SalaryLevel"] = pd.cut(
    df["MonthlyIncome"],
    bins=[0, 4000, 8000, 12000, np.inf],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# Employee tenure category
df["TenureLevel"] = pd.cut(
    df["YearsAtCompany"],
    bins=[-1, 2, 5, 10, np.inf],
    labels=[
        "New",
        "Short Term",
        "Experienced",
        "Long Term"
    ]
)

# Overall satisfaction
df["OverallSatisfaction"] = (
    df["JobSatisfaction"]
    + df["EnvironmentSatisfaction"]
    + df["RelationshipSatisfaction"]
    + df["WorkLifeBalance"]
) / 4

# Career growth
df["CareerGrowthScore"] = (
    df["JobLevel"]
    + df["YearsInCurrentRole"]
    + df["YearsSinceLastPromotion"]
)

# Company experience ratio
df["CompanyExperienceRatio"] = (
    df["YearsAtCompany"]
    /
    (df["TotalWorkingYears"] + 1)
)


# =========================================================
# 5. ONE-HOT ENCODING
# =========================================================

df = pd.get_dummies(
    df,
    drop_first=True
)


# =========================================================
# 6. SEPARATE X AND Y
# =========================================================

X = df.drop(
    "Attrition",
    axis=1
)

y = df["Attrition"]


print("\nNumber of Features:")
print(X.shape[1])


# =========================================================
# 7. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# =========================================================
# 8. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# 9. HANDLE CLASS IMBALANCE USING SMOTE
# =========================================================

print("\nClass Distribution Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nClass Distribution After SMOTE:")
print(
    pd.Series(y_train_smote).value_counts()
)


# =========================================================
# 10. LOGISTIC REGRESSION
# =========================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

lr_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

lr_model.fit(
    X_train_smote,
    y_train_smote
)

lr_prediction = lr_model.predict(
    X_test_scaled
)

lr_probability = lr_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# 11. RANDOM FOREST
# =========================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

rf_prediction = rf_model.predict(
    X_test_scaled
)

rf_probability = rf_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# 12. MODEL EVALUATION FUNCTION
# =========================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall_Sensitivity": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }


# =========================================================
# 13. CALCULATE MODEL METRICS
# =========================================================

lr_metrics = calculate_metrics(
    y_test,
    lr_prediction,
    lr_probability
)

rf_metrics = calculate_metrics(
    y_test,
    rf_prediction,
    rf_probability
)


# =========================================================
# 14. MODEL COMPARISON
# =========================================================

comparison = pd.DataFrame(
    [
        lr_metrics,
        rf_metrics
    ],
    index=[
        "Logistic Regression",
        "Random Forest"
    ]
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    comparison.round(4)
)


# =========================================================
# 15. CREATE RESULTS FOLDER
# =========================================================

Path("results").mkdir(
    exist_ok=True
)

Path("models").mkdir(
    exist_ok=True
)


# =========================================================
# 16. SAVE MODEL COMPARISON
# =========================================================

comparison.to_csv(
    "results/model_comparison.csv"
)


# =========================================================
# 17. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)


print("\nTop 15 Important Features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# =========================================================
# 18. SAVE MODELS
# =========================================================

joblib.dump(
    lr_model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)


# =========================================================
# 19. SAVE TEST PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame({
    "Actual_Attrition": y_test.values,

    "Logistic_Regression_Prediction":
        lr_prediction,

    "Logistic_Regression_Probability":
        lr_probability,

    "Random_Forest_Prediction":
        rf_prediction,

    "Random_Forest_Probability":
        rf_probability
})

predictions_df.to_csv(
    "results/predictions.csv",
    index=False
)


# =========================================================
# 20. FINAL MESSAGE
# =========================================================

print("\n" + "=" * 70)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print("✓ results/model_comparison.csv")
print("✓ results/feature_importance.csv")
print("✓ results/predictions.csv")

print("✓ models/logistic_regression.pkl")
print("✓ models/random_forest.pkl")
print("✓ models/scaler.pkl")
print("✓ models/feature_columns.pkl")

print("\nBoth classification models are ready.")