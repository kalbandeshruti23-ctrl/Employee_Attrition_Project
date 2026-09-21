import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="🤎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BROWN + CREAM THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F7F0E6;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #4A2C20;
}

[data-testid="stSidebar"] * {
    color: #FFF8ED !important;
}

/* MAIN TITLE */
.main-title {
    background-color: #5A3425;
    padding: 25px;
    border-radius: 18px;
    color: #FFF8ED;
    text-align: center;
    font-size: 38px;
    font-weight: 900;
    box-shadow: 0px 8px 20px rgba(74,44,32,0.20);
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #6B4738;
    font-size: 18px;
    font-weight: 800;
    margin: 15px 0 30px 0;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #4A2C20 !important;
    font-weight: 900 !important;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background-color: #FFF8ED;
    border: 2px solid #D7BFA9;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(74,44,32,0.10);
}

div[data-testid="stMetricLabel"] {
    color: #6B4738 !important;
    font-weight: 900 !important;
}

div[data-testid="stMetricValue"] {
    color: #4A2C20 !important;
    font-weight: 900 !important;
}

/* BUTTON */
.stButton > button {
    background-color: #6B3F2F;
    color: #FFF8ED;
    border: none;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 900;
    padding: 12px;
}

.stButton > button:hover {
    background-color: #4A2C20;
    color: #FFF8ED;
}

/* INFO BOX */
.stAlert {
    border-radius: 12px;
}

/* PREDICTION */
.prediction-high {
    background-color: #E8CFC0;
    border: 3px solid #7A3E2D;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    color: #4A2C20;
}

.prediction-low {
    background-color: #E7E2D4;
    border: 3px solid #65704D;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    color: #4A2C20;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 2px solid #D7BFA9;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/employee_attrition_2000_IBM_structure.csv"

MODEL_PATH = "models/tuned_random_forest.pkl"

SCALER_PATH = "models/scaler.pkl"

FEATURE_PATH = "models/feature_columns.pkl"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURE_PATH)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    df = load_data()

    model = load_model()

    scaler = load_scaler()

    feature_columns = load_features()

except Exception as e:

    st.error("❌ Project files could not be loaded.")

    st.write("Please check that these files exist:")

    st.code("""
data/
    employee_attrition_2000_IBM_structure.csv

models/
    tuned_random_forest.pkl
    scaler.pkl
    feature_columns.pkl
""")

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🤎 Employee Attrition Analytics & Prediction
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Classification • Employee Insights • Risk Prediction • Data Visualization
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🤎 Navigation")

page = st.sidebar.radio(
    "Choose Dashboard Section",
    [
        "🏠 Overview",
        "📊 Employee Analysis",
        "🤖 Model Performance",
        "🔮 Attrition Prediction",
        "⭐ Feature Importance",
        "📋 Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### Project

    **Predicting Employee Attrition Using Classification Models**

    **Models**
    - Logistic Regression
    - Random Forest

    **Techniques**
    - Data Cleaning
    - Feature Engineering
    - Encoding
    - Scaling
    - SMOTE
    """
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("📌 Project Overview")

    total_employees = len(df)

    employees_left = int(
        (df["Attrition"] == "Yes").sum()
    )

    employees_retained = int(
        (df["Attrition"] == "No").sum()
    )

    attrition_rate = (
        employees_left / total_employees
    ) * 100


    # METRICS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Total Employees",
        f"{total_employees:,}"
    )

    c2.metric(
        "🚪 Employees Left",
        f"{employees_left:,}"
    )

    c3.metric(
        "✅ Employees Retained",
        f"{employees_retained:,}"
    )

    c4.metric(
        "📉 Attrition Rate",
        f"{attrition_rate:.2f}%"
    )


    st.markdown("---")

    # CHARTS

    c1, c2 = st.columns(2)


    with c1:

        st.subheader("📊 Attrition Distribution")

        counts = df["Attrition"].value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            counts.index,
            counts.values
        )

        ax.set_xlabel("Attrition")

        ax.set_ylabel("Number of Employees")

        ax.set_title(
            "Employee Attrition Distribution",
            fontweight="bold"
        )

        fig.tight_layout()

        st.pyplot(fig)


    with c2:

        st.subheader("🏢 Department Distribution")

        counts = df["Department"].value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            counts.index,
            counts.values
        )

        ax.set_xlabel("Department")

        ax.set_ylabel("Employees")

        ax.set_title(
            "Department-wise Employee Distribution",
            fontweight="bold"
        )

        plt.xticks(rotation=20)

        fig.tight_layout()

        st.pyplot(fig)


    st.markdown("---")

    st.subheader("💡 Key Employee Insights")

    overtime_rate = (
        df.groupby("OverTime")["Attrition"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📉 Overall Attrition",
        f"{attrition_rate:.2f}%"
    )

    c2.metric(
        "⏰ Overtime Attrition",
        f"{overtime_rate.get('Yes', 0):.2f}%"
    )

    c3.metric(
        "📋 Dataset Records",
        f"{total_employees:,}"
    )


# ============================================================
# PAGE 2 — EMPLOYEE ANALYSIS
# ============================================================

elif page == "📊 Employee Analysis":

    st.header("📊 Exploratory Employee Analysis")


    tab1, tab2, tab3 = st.tabs(
        [
            "⏰ Overtime",
            "😊 Satisfaction",
            "💰 Salary & Tenure"
        ]
    )


    # OVERTIME

    with tab1:

        st.subheader(
            "⏰ Overtime vs Employee Attrition"
        )

        table = pd.crosstab(
            df["OverTime"],
            df["Attrition"]
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        table.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("OverTime")

        ax.set_ylabel("Employees")

        ax.set_title(
            "Overtime vs Attrition",
            fontweight="bold"
        )

        plt.xticks(rotation=0)

        fig.tight_layout()

        st.pyplot(fig)


    # SATISFACTION

    with tab2:

        st.subheader(
            "😊 Job Satisfaction vs Attrition"
        )

        table = pd.crosstab(
            df["JobSatisfaction"],
            df["Attrition"]
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        table.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Job Satisfaction"
        )

        ax.set_ylabel(
            "Employees"
        )

        ax.set_title(
            "Job Satisfaction vs Attrition",
            fontweight="bold"
        )

        plt.xticks(rotation=0)

        fig.tight_layout()

        st.pyplot(fig)


        st.subheader(
            "⚖️ Work-Life Balance vs Attrition"
        )

        table = pd.crosstab(
            df["WorkLifeBalance"],
            df["Attrition"]
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        table.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Work-Life Balance"
        )

        ax.set_ylabel(
            "Employees"
        )

        ax.set_title(
            "Work-Life Balance vs Attrition",
            fontweight="bold"
        )

        plt.xticks(rotation=0)

        fig.tight_layout()

        st.pyplot(fig)


    # SALARY AND TENURE

    with tab3:

        c1, c2 = st.columns(2)


        with c1:

            st.subheader(
                "💰 Monthly Income"
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.boxplot(
                data=df,
                x="Attrition",
                y="MonthlyIncome",
                ax=ax
            )

            ax.set_title(
                "Monthly Income vs Attrition",
                fontweight="bold"
            )

            fig.tight_layout()

            st.pyplot(fig)


        with c2:

            st.subheader(
                "🏢 Years at Company"
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.boxplot(
                data=df,
                x="Attrition",
                y="YearsAtCompany",
                ax=ax
            )

            ax.set_title(
                "Years at Company vs Attrition",
                fontweight="bold"
            )

            fig.tight_layout()

            st.pyplot(fig)


    st.subheader(
        "🎂 Age vs Attrition"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.boxplot(
        data=df,
        x="Attrition",
        y="Age",
        ax=ax
    )

    ax.set_title(
        "Age vs Employee Attrition",
        fontweight="bold"
    )

    fig.tight_layout()

    st.pyplot(fig)


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Machine Learning Model Performance"
    )


    comparison_path = (
        "results/model_comparison.csv"
    )


    if os.path.exists(comparison_path):

        comparison_df = pd.read_csv(
            comparison_path
        )


        # ----------------------------------------------------
        # FIX MODEL COLUMN
        # ----------------------------------------------------

        if "Model" in comparison_df.columns:

            model_column = "Model"

        elif "Unnamed: 0" in comparison_df.columns:

            model_column = "Unnamed: 0"

        else:

            model_column = comparison_df.columns[0]


        # ----------------------------------------------------
        # STANDARDIZE DISPLAY
        # ----------------------------------------------------

        display_df = comparison_df.copy()

        display_df = display_df.rename(
            columns={
                model_column: "Model",
                "F1_Score": "F1 Score",
                "ROC_AUC": "ROC-AUC",
                "Recall_Sensitivity": "Recall"
            }
        )


        st.subheader(
            "📊 Model Comparison"
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )


        st.markdown("---")

        st.subheader(
            "📌 Performance Metrics"
        )


        # TWO MODEL CARDS

        metric_columns = st.columns(
            min(2, len(display_df))
        )


        for i, row in display_df.iterrows():

            with metric_columns[
                i % len(metric_columns)
            ]:

                model_name = row["Model"]

                st.markdown(
                    f"### 🤖 {model_name}"
                )


                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Accuracy",
                        f"{float(row['Accuracy']):.3f}"
                    )

                    st.metric(
                        "Precision",
                        f"{float(row['Precision']):.3f}"
                    )


                with c2:

                    st.metric(
                        "Recall",
                        f"{float(row['Recall']):.3f}"
                    )

                    st.metric(
                        "F1 Score",
                        f"{float(row['F1 Score']):.3f}"
                    )


                st.metric(
                    "ROC-AUC",
                    f"{float(row['ROC-AUC']):.3f}"
                )


    else:

        st.warning(
            "model_comparison.csv was not found."
        )


    st.markdown("---")


    # CONFUSION MATRICES

    st.subheader(
        "🔲 Confusion Matrices"
    )

    c1, c2 = st.columns(2)


    logistic_path = (
        "visualizations/"
        "confusion_matrix_logistic.png"
    )

    rf_path = (
        "visualizations/"
        "confusion_matrix_random_forest.png"
    )


    with c1:

        st.markdown(
            "### Logistic Regression"
        )

        if os.path.exists(logistic_path):

            st.image(
                logistic_path,
                use_container_width=True
            )

        else:

            st.warning(
                "Logistic Regression confusion matrix not found."
            )


    with c2:

        st.markdown(
            "### Random Forest"
        )

        if os.path.exists(rf_path):

            st.image(
                rf_path,
                use_container_width=True
            )

        else:

            st.warning(
                "Random Forest confusion matrix not found."
            )


    # ROC CURVE

    st.subheader(
        "📈 ROC Curve Comparison"
    )

    roc_path = (
        "visualizations/"
        "roc_curve_comparison.png"
    )


    if os.path.exists(roc_path):

        st.image(
            roc_path,
            use_container_width=True
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


# ============================================================
# PAGE 4 — ATTRITION PREDICTION
# ============================================================

elif page == "🔮 Attrition Prediction":

    st.header(
        "🔮 Individual Employee Attrition Prediction"
    )

    st.info(
        "Enter employee details and click the prediction button."
    )


    # --------------------------------------------------------
    # NUMERIC / RATING INPUTS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        Age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        DailyRate = st.number_input(
            "Daily Rate",
            min_value=100,
            max_value=1500,
            value=700
        )

        DistanceFromHome = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=30,
            value=5
        )

        Education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5],
            index=2
        )

        EnvironmentSatisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4],
            index=2
        )

        HourlyRate = st.number_input(
            "Hourly Rate",
            min_value=30,
            max_value=100,
            value=65
        )

        JobInvolvement = st.selectbox(
            "Job Involvement",
            [1, 2, 3, 4],
            index=2
        )

        JobLevel = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5],
            index=0
        )


    with c2:

        JobSatisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4],
            index=2
        )

        MonthlyIncome = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=100000,
            value=5000
        )

        MonthlyRate = st.number_input(
            "Monthly Rate",
            min_value=2000,
            max_value=30000,
            value=14000
        )

        NumCompaniesWorked = st.number_input(
            "Companies Worked",
            min_value=0,
            max_value=10,
            value=2
        )

        PercentSalaryHike = st.number_input(
            "Percent Salary Hike",
            min_value=10,
            max_value=30,
            value=15
        )

        PerformanceRating = st.selectbox(
            "Performance Rating",
            [1, 2, 3, 4],
            index=1
        )

        RelationshipSatisfaction = st.selectbox(
            "Relationship Satisfaction",
            [1, 2, 3, 4],
            index=2
        )

        StockOptionLevel = st.selectbox(
            "Stock Option Level",
            [0, 1, 2, 3],
            index=0
        )


    with c3:

        TotalWorkingYears = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=50,
            value=8
        )

        TrainingTimesLastYear = st.number_input(
            "Training Times Last Year",
            min_value=0,
            max_value=10,
            value=3
        )

        WorkLifeBalance = st.selectbox(
            "Work-Life Balance",
            [1, 2, 3, 4],
            index=2
        )

        YearsAtCompany = st.number_input(
            "Years At Company",
            min_value=0,
            max_value=50,
            value=5
        )

        YearsInCurrentRole = st.number_input(
            "Years In Current Role",
            min_value=0,
            max_value=20,
            value=3
        )

        YearsSinceLastPromotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            max_value=15,
            value=1
        )

        YearsWithCurrManager = st.number_input(
            "Years With Current Manager",
            min_value=0,
            max_value=20,
            value=3
        )


    # --------------------------------------------------------
    # CATEGORICAL INPUTS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🏢 Job & Personal Information"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        BusinessTravel = st.selectbox(
            "Business Travel",
            [
                "Non-Travel",
                "Travel_Rarely",
                "Travel_Frequently"
            ]
        )

        Department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources"
            ]
        )

        EducationField = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )


    with c2:

        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        JobRole = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )

        MaritalStatus = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Divorced"
            ]
        )


    with c3:

        OverTime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    st.markdown("---")

    predict = st.button(
        "🔮 PREDICT EMPLOYEE ATTRITION",
        use_container_width=True
    )


    if predict:

        # ----------------------------------------------------
        # CREATE EMPLOYEE DATAFRAME
        # ----------------------------------------------------

        employee = pd.DataFrame({

            "Age": [Age],

            "BusinessTravel": [BusinessTravel],

            "DailyRate": [DailyRate],

            "Department": [Department],

            "DistanceFromHome": [DistanceFromHome],

            "Education": [Education],

            "EducationField": [EducationField],

            "EnvironmentSatisfaction":
                [EnvironmentSatisfaction],

            "Gender": [Gender],

            "HourlyRate": [HourlyRate],

            "JobInvolvement": [JobInvolvement],

            "JobLevel": [JobLevel],

            "JobRole": [JobRole],

            "JobSatisfaction": [JobSatisfaction],

            "MaritalStatus": [MaritalStatus],

            "MonthlyIncome": [MonthlyIncome],

            "MonthlyRate": [MonthlyRate],

            "NumCompaniesWorked":
                [NumCompaniesWorked],

            "PercentSalaryHike":
                [PercentSalaryHike],

            "PerformanceRating":
                [PerformanceRating],

            "RelationshipSatisfaction":
                [RelationshipSatisfaction],

            "StockOptionLevel":
                [StockOptionLevel],

            "TotalWorkingYears":
                [TotalWorkingYears],

            "TrainingTimesLastYear":
                [TrainingTimesLastYear],

            "WorkLifeBalance":
                [WorkLifeBalance],

            "YearsAtCompany":
                [YearsAtCompany],

            "YearsInCurrentRole":
                [YearsInCurrentRole],

            "YearsSinceLastPromotion":
                [YearsSinceLastPromotion],

            "YearsWithCurrManager":
                [YearsWithCurrManager],

            "OverTime": [OverTime]
        })


        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        employee["SalaryLevel"] = pd.cut(
            employee["MonthlyIncome"],
            bins=[
                0,
                3000,
                7000,
                np.inf
            ],
            labels=[
                "Low",
                "Medium",
                "High"
            ]
        )


        employee["TenureLevel"] = pd.cut(
            employee["YearsAtCompany"],
            bins=[
                -1,
                2,
                7,
                np.inf
            ],
            labels=[
                "New",
                "Mid",
                "Experienced"
            ]
        )


        employee["OverallSatisfaction"] = (

            employee["JobSatisfaction"]

            + employee[
                "EnvironmentSatisfaction"
            ]

            + employee[
                "RelationshipSatisfaction"
            ]

            + employee[
                "WorkLifeBalance"
            ]

        ) / 4


        employee["CareerGrowthScore"] = (

            employee["JobLevel"]

            + employee["YearsAtCompany"]

            + employee[
                "TrainingTimesLastYear"
            ]

        )


        employee["CompanyExperienceRatio"] = (

            employee["YearsAtCompany"]

            /

            employee[
                "TotalWorkingYears"
            ].replace(0, 1)

        )


        # ----------------------------------------------------
        # ONE HOT ENCODING
        # ----------------------------------------------------

        encoded = pd.get_dummies(
            employee,
            drop_first=False
        )


        # MATCH TRAINING FEATURES

        encoded = encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        scaled = scaler.transform(
            encoded
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(scaled)[0]
        )


        probability = float(
            model.predict_proba(
                scaled
            )[0][1]
        )


        probability_percent = (
            probability * 100
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🎯 Prediction Result"
        )


        if prediction == 1:

            st.markdown(
                f"""
                <div class="prediction-high">

                <h2>⚠️ HIGH ATTRITION RISK</h2>

                <h3>
                {probability_percent:.2f}%
                </h3>

                <p>
                Estimated probability of employee attrition
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            result_text = "Attrition"


        else:

            st.markdown(
                f"""
                <div class="prediction-low">

                <h2>✅ LOW ATTRITION RISK</h2>

                <h3>
                {probability_percent:.2f}%
                </h3>

                <p>
                Estimated probability of employee attrition
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            result_text = "No Attrition"


        # PROBABILITY

        st.markdown(
            "### 📊 Attrition Probability"
        )

        st.progress(
            probability
        )


        c1, c2 = st.columns(2)

        c1.metric(
            "Predicted Class",
            result_text
        )

        c2.metric(
            "Attrition Probability",
            f"{probability_percent:.2f}%"
        )


# ============================================================
# PAGE 5 — FEATURE IMPORTANCE
# ============================================================

elif page == "⭐ Feature Importance":

    st.header(
        "⭐ Random Forest Feature Importance"
    )


    path = (
        "results/feature_importance.csv"
    )


    if os.path.exists(path):

        importance = pd.read_csv(
            path
        )


        importance = importance.sort_values(
            by="Importance",
            ascending=False
        ).head(15)


        st.subheader(
            "🌟 Top 15 Important Features"
        )


        st.dataframe(
            importance,
            use_container_width=True
        )


        fig, ax = plt.subplots(
            figsize=(11, 7)
        )


        ax.barh(
            importance["Feature"][::-1],
            importance["Importance"][::-1]
        )


        ax.set_xlabel(
            "Importance"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Top Features Influencing Employee Attrition",
            fontweight="bold"
        )


        fig.tight_layout()

        st.pyplot(fig)


        st.info(
            "Feature importance shows which variables were "
            "used strongly by the Random Forest model. "
            "It does not by itself prove causation."
        )


    else:

        st.warning(
            "feature_importance.csv was not found."
        )


# ============================================================
# PAGE 6 — DATASET
# ============================================================

elif page == "📋 Dataset":

    st.header(
        "📋 Employee Attrition Dataset"
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )


    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df,
        use_container_width=True,
        height=550
    )


    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Dataset CSV",
        data=csv_data,
        file_name="employee_attrition_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6B4738;
        font-weight:900;
        font-size:16px;
    ">

    🤎 Employee Attrition Analytics & Prediction System

    <br>

    Logistic Regression • Random Forest • SMOTE • Data Visualization

    </div>
    """,
    unsafe_allow_html=True
)