import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# =========================================================
# 1. LOAD DATASET
# =========================================================

DATA_PATH = "data/employee_attrition_2000_IBM_structure.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("MODEL EVALUATION AND VISUALIZATION")
print("=" * 70)


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

df["OverallSatisfaction"] = (
    df["JobSatisfaction"]
    + df["EnvironmentSatisfaction"]
    + df["RelationshipSatisfaction"]
    + df["WorkLifeBalance"]
) / 4

df["CareerGrowthScore"] = (
    df["JobLevel"]
    + df["YearsInCurrentRole"]
    + df["YearsSinceLastPromotion"]
)

df["CompanyExperienceRatio"] = (
    df["YearsAtCompany"]
    /
    (df["TotalWorkingYears"] + 1)
)


# =========================================================
# 5. ENCODE CATEGORICAL VARIABLES
# =========================================================

df = pd.get_dummies(
    df,
    drop_first=True
)


# =========================================================
# 6. FEATURES AND TARGET
# =========================================================

X = df.drop(
    "Attrition",
    axis=1
)

y = df["Attrition"]


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


# =========================================================
# 8. SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# 9. SMOTE
# =========================================================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)


# =========================================================
# 10. TRAIN LOGISTIC REGRESSION
# =========================================================

lr_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

lr_model.fit(
    X_train_smote,
    y_train_smote
)

lr_pred = lr_model.predict(
    X_test_scaled
)

lr_prob = lr_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# 11. TRAIN RANDOM FOREST
# =========================================================

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

rf_pred = rf_model.predict(
    X_test_scaled
)

rf_prob = rf_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# 12. CREATE VISUALIZATION FOLDER
# =========================================================

Path("visualizations").mkdir(
    exist_ok=True
)


# =========================================================
# 13. CONFUSION MATRIX - LOGISTIC REGRESSION
# =========================================================

cm_lr = confusion_matrix(
    y_test,
    lr_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "visualizations/confusion_matrix_logistic.png",
    dpi=300
)

plt.show()


# =========================================================
# 14. CONFUSION MATRIX - RANDOM FOREST
# =========================================================

cm_rf = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title(
    "Random Forest - Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "visualizations/confusion_matrix_random_forest.png",
    dpi=300
)

plt.show()


# =========================================================
# 15. CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        lr_pred,
        target_names=[
            "Stay",
            "Leave"
        ],
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "Stay",
            "Leave"
        ],
        zero_division=0
    )
)


# =========================================================
# 16. ROC CURVE
# =========================================================

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    lr_prob
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    rf_prob
)

auc_lr = roc_auc_score(
    y_test,
    lr_prob
)

auc_rf = roc_auc_score(
    y_test,
    rf_prob
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {auc_lr:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Model Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "visualizations/roc_curve_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top_features = feature_importance.head(15)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Factors Influencing Employee Attrition"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    "visualizations/random_forest_feature_importance.png",
    dpi=300
)

plt.show()


# =========================================================
# 18. MODEL PERFORMANCE COMPARISON
# =========================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "ROC_AUC": [
        auc_lr,
        auc_rf
    ]
})


plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="ROC_AUC"
)

plt.title(
    "ROC-AUC Model Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "ROC-AUC"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "visualizations/model_roc_auc_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 19. CROSS VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

rf_cv_scores = cross_val_score(
    rf_model,
    X_train_smote,
    y_train_smote,
    cv=5,
    scoring="f1"
)

lr_cv_scores = cross_val_score(
    lr_model,
    X_train_smote,
    y_train_smote,
    cv=5,
    scoring="f1"
)

print("\nLogistic Regression CV F1 Scores:")
print(lr_cv_scores)

print(
    "Logistic Regression Mean F1:",
    round(lr_cv_scores.mean(), 4)
)

print("\nRandom Forest CV F1 Scores:")
print(rf_cv_scores)

print(
    "Random Forest Mean F1:",
    round(rf_cv_scores.mean(), 4)
)


# =========================================================
# 20. RANDOM FOREST HYPERPARAMETER TUNING
# =========================================================

print("\n" + "=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

param_grid = {
    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        5,
        10,
        15
    ],

    "min_samples_split": [
        2,
        5
    ]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1
)

grid_search.fit(
    X_train_smote,
    y_train_smote
)

print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)

print(
    "\nBest Cross-Validation F1:",
    round(
        grid_search.best_score_,
        4
    )
)


# =========================================================
# 21. SAVE TUNED MODEL
# =========================================================

best_rf = grid_search.best_estimator_

joblib.dump(
    best_rf,
    "models/tuned_random_forest.pkl"
)


# =========================================================
# 22. SAVE FEATURE IMPORTANCE
# =========================================================

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)


# =========================================================
# 23. FINAL MESSAGE
# =========================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)

print("\nVisualizations created:")

print("✓ confusion_matrix_logistic.png")
print("✓ confusion_matrix_random_forest.png")
print("✓ roc_curve_comparison.png")
print("✓ random_forest_feature_importance.png")
print("✓ model_roc_auc_comparison.png")

print("\nTuned model created:")
print("✓ models/tuned_random_forest.pkl")