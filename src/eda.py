import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

DATA_PATH = "data/employee_attrition_2000_IBM_structure.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())


# ---------------------------------------------------------
# 2. CREATE VISUALIZATION FOLDER
# ---------------------------------------------------------

Path("visualizations").mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# 3. ATTRITION DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title(
    "Employee Attrition Distribution"
)

plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_distribution.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 4. ATTRITION BY OVERTIME
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title(
    "Employee Attrition by Overtime"
)

plt.xlabel("Overtime")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_overtime.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 5. ATTRITION BY JOB SATISFACTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)

plt.title(
    "Employee Attrition by Job Satisfaction"
)

plt.xlabel("Job Satisfaction")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_job_satisfaction.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 6. ATTRITION BY WORK-LIFE BALANCE
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="WorkLifeBalance",
    hue="Attrition"
)

plt.title(
    "Employee Attrition by Work-Life Balance"
)

plt.xlabel("Work-Life Balance")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_worklife_balance.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 7. ATTRITION BY DEPARTMENT
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)

plt.title(
    "Employee Attrition by Department"
)

plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_department.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 8. MONTHLY INCOME VS ATTRITION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title(
    "Monthly Income vs Employee Attrition"
)

plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.tight_layout()

plt.savefig(
    "visualizations/income_vs_attrition.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 9. YEARS AT COMPANY VS ATTRITION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany"
)

plt.title(
    "Years at Company vs Employee Attrition"
)

plt.xlabel("Attrition")
plt.ylabel("Years at Company")

plt.tight_layout()

plt.savefig(
    "visualizations/years_at_company.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 10. AGE VS ATTRITION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="Age"
)

plt.title(
    "Age vs Employee Attrition"
)

plt.xlabel("Attrition")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    "visualizations/age_vs_attrition.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 11. JOB ROLE VS ATTRITION
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="JobRole",
    hue="Attrition"
)

plt.title(
    "Employee Attrition by Job Role"
)

plt.xlabel("Job Role")
plt.ylabel("Number of Employees")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_job_role.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 12. CORRELATION HEATMAP
# ---------------------------------------------------------

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

plt.figure(
    figsize=(16, 12)
)

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Heatmap of Employee Attributes"
)

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 13. ATTRITION RATE BY OVERTIME
# ---------------------------------------------------------

overtime_attrition = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Overtime (%):")
print(overtime_attrition)


# ---------------------------------------------------------
# 14. ATTRITION RATE BY JOB SATISFACTION
# ---------------------------------------------------------

satisfaction_attrition = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Job Satisfaction (%):")
print(satisfaction_attrition)


# ---------------------------------------------------------
# 15. ATTRITION RATE BY DEPARTMENT
# ---------------------------------------------------------

department_attrition = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Department (%):")
print(department_attrition)


# ---------------------------------------------------------
# 16. FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nVisualizations saved in:")
print("visualizations/")