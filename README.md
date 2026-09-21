# Predicting Employee Attrition Using Classification Models

## 📌 Project Overview

**Predicting Employee Attrition Using Classification Models** is a machine learning project developed to predict whether an employee is likely to leave an organization based on various employee-related factors.

The project analyzes factors such as **job satisfaction, monthly income, years at company, work-life balance, overtime, age, department, and job role**. Classification algorithms including **Logistic Regression and Random Forest** are used to build predictive models.

The project provides a complete machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, visualization, and employee attrition prediction.

---

## 🎯 Objectives

1. To analyze employee data and identify factors associated with employee attrition.
2. To preprocess and prepare employee data for machine learning.
3. To perform exploratory data analysis using meaningful visualizations.
4. To implement Logistic Regression and Random Forest classification models.
5. To evaluate and compare the performance of the classification models.
6. To develop a system that can predict potential employee attrition.

---

## 📝 Abstract

Employee attrition is an important challenge for organizations because frequent employee turnover can affect productivity, workforce stability, and organizational planning. This project applies machine learning classification techniques to predict employee attrition using employee-related attributes.

The dataset is preprocessed by handling categorical variables and preparing relevant features for machine learning. Exploratory Data Analysis is performed to understand relationships between employee characteristics and attrition. Logistic Regression and Random Forest models are trained and evaluated using the prepared dataset. Performance evaluation is performed using classification metrics and confusion matrices. Feature-importance analysis is also used to understand which attributes contribute to the model's predictions.

The developed system demonstrates how machine learning can be applied to Human Resource Analytics for analyzing employee attrition patterns and supporting data-driven workforce planning.

---

## 🛠️ Technologies Used

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Programming language                |
| Pandas       | Data processing and analysis        |
| NumPy        | Numerical operations                |
| Scikit-learn | Machine learning                    |
| Matplotlib   | Data visualization                  |
| Seaborn      | Statistical visualization           |
| Joblib       | Saving trained models               |
| Streamlit    | Interactive application             |
| VS Code      | Development environment             |
| Git & GitHub | Version control and project hosting |

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as a classification algorithm to predict employee attrition based on selected employee features.

### 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification. It is also used to determine the relative importance of different features.

---

## 🔄 Methodology

The project follows these major steps:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
Model Evaluation
   ↓
Feature Importance Analysis
   ↓
Attrition Prediction
```

---

## 📊 Dataset

The project uses an employee attrition dataset containing employee-related information.

Important attributes include:

* Age
* Department
* Job Role
* Job Satisfaction
* Monthly Income
* Years at Company
* Work-Life Balance
* Overtime
* Education
* Environment Satisfaction
* Job Involvement
* Career-related attributes
* Attrition

The prepared dataset is available in:

```text
data/employee_attrition_2000_IBM_structure.csv
```

---

## 🔍 Exploratory Data Analysis

Several visualizations are generated to understand employee attrition patterns.

The project includes visualizations for:

* Attrition Distribution
* Attrition by Department
* Attrition by Job Role
* Job Satisfaction vs Attrition
* Overtime vs Attrition
* Work-Life Balance vs Attrition
* Income vs Attrition
* Age vs Attrition
* Years at Company
* Correlation Heatmap

---

## 📈 Model Evaluation

The trained models are evaluated using:

* Accuracy
* Confusion Matrix
* ROC Curve
* ROC-AUC comparison
* Model comparison
* Feature importance

The generated results are stored in:

```text
results/
```

Important result files include:

```text
model_comparison.csv
predictions.csv
feature_importance.csv
processed_employee_data.csv
```

---

## 📁 Project Structure

```text
Employee_Attrition_Project/
│
├── data/
│   └── employee_attrition_2000_IBM_structure.csv
│
├── models/
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── tuned_random_forest.pkl
│   └── scaler.pkl
│
├── results/
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   ├── predictions.csv
│   └── processed_employee_data.csv
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── visualizations/
│   ├── attrition_distribution.png
│   ├── attrition_department.png
│   ├── attrition_job_role.png
│   ├── attrition_job_satisfaction.png
│   ├── attrition_overtime.png
│   ├── attrition_worklife_balance.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_random_forest.png
│   ├── model_roc_auc_comparison.png
│   ├── random_forest_feature_importance.png
│   └── roc_curve_comparison.png
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/kalbandeshruti23-ctrl/Employee_Attrition_Project.git
```

### Step 2: Open the Project

```bash
cd Employee_Attrition_Project
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

For Windows:

```powershell
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Main Program

```bash
python main.py
```

### Step 7: Run the Streamlit Application

```bash
streamlit run app.py
```

The interactive employee attrition prediction dashboard will open in the browser.

---

## 💡 Key Features

* Employee attrition prediction
* Data preprocessing
* Exploratory Data Analysis
* Logistic Regression classification
* Random Forest classification
* Model comparison
* Confusion matrix visualization
* ROC-AUC analysis
* Feature importance analysis
* Interactive Streamlit application
* Saved trained machine learning models

---

## 📌 Results and Discussion

The developed system demonstrates the practical application of machine learning in Human Resource Analytics. Logistic Regression and Random Forest models are trained using employee-related features and evaluated on test data.

The generated confusion matrices and ROC curves provide an understanding of model performance, while feature-importance analysis helps identify employee attributes that contribute to the model's predictions.

The predictions should be interpreted as **machine-learning estimates based on the available dataset**, rather than certain predictions about an individual employee's future decision.

---

## 🔮 Future Scope

The project can be further enhanced by:

* Using larger and more diverse employee datasets.
* Applying advanced classification algorithms.
* Performing hyperparameter optimization.
* Adding explainable AI techniques.
* Developing a real-time HR analytics dashboard.
* Adding employee retention recommendations.
* Deploying the application on a cloud platform.
* Integrating the system with organizational HR databases.

---

## 👩‍💻 Author

**Shruti Kalbande**

M.Tech – Computer Science / Big Data Analytics

---

## 📄 License

This project is developed for **academic, learning, and internship purposes**.
