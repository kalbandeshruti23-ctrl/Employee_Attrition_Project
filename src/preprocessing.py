import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

DATA_PATH = "data/employee_attrition_2000_IBM_structure.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("EMPLOYEE ATTRITION DATASET")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ---------------------------------------------------------
# 2. CHECK DATA INFORMATION
# ---------------------------------------------------------

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())


# ---------------------------------------------------------
# 3. REMOVE UNNECESSARY COLUMNS
# ---------------------------------------------------------

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

print("\nColumns after removing unnecessary columns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# ---------------------------------------------------------
# 5. CONVERT TARGET VARIABLE
# ---------------------------------------------------------

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print("\nTarget variable after encoding:")
print(df["Attrition"].value_counts())


# ---------------------------------------------------------
# 6. FEATURE ENGINEERING
# ---------------------------------------------------------

# Salary category
df["SalaryLevel"] = pd.cut(
    df["MonthlyIncome"],
    bins=[0, 4000, 8000, 12000, np.inf],
    labels=["Low", "Medium", "High", "Very High"]
)

# Tenure category
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

# Satisfaction score
df["OverallSatisfaction"] = (
    df["JobSatisfaction"]
    + df["EnvironmentSatisfaction"]
    + df["RelationshipSatisfaction"]
    + df["WorkLifeBalance"]
) / 4

# Career growth indicator
df["CareerGrowthScore"] = (
    df["JobLevel"]
    + df["YearsInCurrentRole"]
    + df["YearsSinceLastPromotion"]
)

# Experience ratio
df["CompanyExperienceRatio"] = (
    df["YearsAtCompany"] /
    (df["TotalWorkingYears"] + 1)
)

print("\nFeature engineering completed.")


# ---------------------------------------------------------
# 7. ONE-HOT ENCODING
# ---------------------------------------------------------

df = pd.get_dummies(
    df,
    drop_first=True
)

print("\nDataset after encoding:")
print(df.shape)


# ---------------------------------------------------------
# 8. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(
    "Attrition",
    axis=1
)

y = df["Attrition"]


# ---------------------------------------------------------
# 9. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ---------------------------------------------------------
# 10. SCALE FEATURES
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 11. HANDLE CLASS IMBALANCE USING SMOTE
# ---------------------------------------------------------

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ---------------------------------------------------------
# 12. SAVE PROCESSED DATA
# ---------------------------------------------------------

processed_data = df.copy()

processed_data.to_csv(
    "results/processed_employee_data.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")


# ---------------------------------------------------------
# 13. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Original Dataset:", (2000, 35))
print("Processed Dataset:", df.shape)
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
print("Features:", X.shape[1])

print("\nReady for model development.")