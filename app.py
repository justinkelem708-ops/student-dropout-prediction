import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Configuration
st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon=None,
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = pickle.load(open('models/logistic_regression_final.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    features = json.load(open('models/feature_names.json', 'r'))
    config = json.load(open('models/config.json', 'r'))
    return model, scaler, features, config

model, scaler, features, config = load_model()

# Header
st.title("Student Dropout Prediction System")
st.markdown(
    "Early warning system for identifying at-risk students "
    "before the first academic evaluations."
)
st.markdown(
    "**Performance** : Recall 80.77% | Precision 44.68% | "
    "F1-Score 0.578 | Threshold 0.399"
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("Model Performance")
    st.metric("Recall", "80.77%", help="Percentage of dropout cases detected")
    st.metric("Precision", "44.68%", help="Alert reliability")
    st.metric("Optimal Threshold", f"{config['optimal_threshold']:.3f}")
    st.markdown("---")
    st.markdown("""
    **Key predictors**
    1. Academic failure history
    2. Social activity frequency
    3. Cumulative support index

    **Design rationale**
    G1 and G2 grades deliberately excluded
    to enable pre-exam early detection.
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Analysis", "Documentation"])

with tab1:
    st.header("Student Profile Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Academic Profile")

        school = st.radio(
            "School",
            ["Gabriel Pereira", "Mousinho da Silveira"],
            horizontal=True
        )
        failures = st.number_input(
            "Past academic failures",
            min_value=0, max_value=4, value=0,
            help="Number of previously failed years"
        )
        absences = st.number_input(
            "Number of absences",
            min_value=0, max_value=93, value=0
        )
        studytime = st.selectbox(
            "Weekly study time",
            ["Less than 2 hours", "2 to 5 hours",
             "5 to 10 hours", "More than 10 hours"]
        )

    with col2:
        st.subheader("Family Context")

        Medu = st.selectbox(
            "Mother's education",
            ["None", "Primary", "Middle school",
             "Secondary", "Higher education"]
        )
        Fedu = st.selectbox(
            "Father's education",
            ["None", "Primary", "Middle school",
             "Secondary", "Higher education"]
        )
        famrel = st.slider(
            "Family relationship quality",
            min_value=1, max_value=5, value=3,
            help="1 = Very poor, 5 = Excellent"
        )
        famsup = st.radio(
            "Family educational support",
            ["No", "Yes"], horizontal=True
        )

    with col3:
        st.subheader("Behavioral Indicators")

        goout = st.slider(
            "Social outing frequency",
            min_value=1, max_value=5, value=3,
            help="1 = Very rare, 5 = Very frequent"
        )
        Dalc = st.slider(
            "Workday alcohol consumption",
            min_value=1, max_value=5, value=1,
            help="1 = Very low, 5 = Very high"
        )
        Walc = st.slider(
            "Weekend alcohol consumption",
            min_value=1, max_value=5, value=1,
            help="1 = Very low, 5 = Very high"
        )
        higher = st.radio(
            "Wants higher education",
            ["Yes", "No"], horizontal=True
        )

    with st.expander("Additional Parameters (optional)"):
        col4, col5 = st.columns(2)
        with col4:
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            age = st.number_input("Age", min_value=15, max_value=22, value=17)
            address = st.radio(
                "Address type", ["Urban", "Rural"], horizontal=True
            )
            Pstatus = st.radio(
                "Parental status", ["Together", "Apart"], horizontal=True
            )
        with col5:
            traveltime = st.selectbox(
                "Travel time to school",
                ["Less than 15 min", "15-30 min",
                 "30 min - 1 hour", "More than 1 hour"]
            )
            activities = st.radio(
                "Extracurricular activities", ["No", "Yes"], horizontal=True
            )
            internet = st.radio(
                "Internet access at home", ["Yes", "No"], horizontal=True
            )
            romantic = st.radio(
                "Romantic relationship", ["No", "Yes"], horizontal=True
            )

    if st.button("Generate Prediction", type="primary", use_container_width=True):

        # Conversions
        school_val = 1 if school == "Gabriel Pereira" else 0
        studytime_val = [
            "Less than 2 hours", "2 to 5 hours",
            "5 to 10 hours", "More than 10 hours"
        ].index(studytime) + 1
        edu_levels = ["None", "Primary", "Middle school",
                      "Secondary", "Higher education"]
        Medu_val = edu_levels.index(Medu)
        Fedu_val = edu_levels.index(Fedu)
        sex_val = 1 if sex == "Female" else 0
        address_val = 1 if address == "Urban" else 0
        Pstatus_val = 1 if Pstatus == "Together" else 0
        traveltime_val = [
            "Less than 15 min", "15-30 min",
            "30 min - 1 hour", "More than 1 hour"
        ].index(traveltime) + 1
        famsup_val = 1 if famsup == "Yes" else 0

        # Engineered features
        parent_edu_mean_val = (Medu_val + Fedu_val) / 2
        total_alcohol_val = Dalc + Walc
        high_risk_val = 1 if (failures > 0 and absences > 10) else 0
        total_support_val = famsup_val

        # Build input
        data = {
            'school': school_val,
            'sex': sex_val,
            'age': age,
            'address': address_val,
            'famsize': 1,
            'Pstatus': Pstatus_val,
            'Medu': Medu_val,
            'Fedu': Fedu_val,
            'Mjob': 0,
            'Fjob': 0,
            'reason': 0,
            'guardian': 0,
            'traveltime': traveltime_val,
            'studytime': studytime_val,
            'failures': failures,
            'schoolsup': 0,
            'famsup': famsup_val,
            'paid': 0,
            'activities': 1 if activities == "Yes" else 0,
            'nursery': 1,
            'higher': 1 if higher == "Yes" else 0,
            'internet': 1 if internet == "Yes" else 0,
            'romantic': 1 if romantic == "Yes" else 0,
            'famrel': famrel,
            'freetime': 3,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': 3,
            'absences': absences,
            'parent_edu_mean': parent_edu_mean_val,
            'total_alcohol': total_alcohol_val,
            'high_risk': high_risk_val,
            'total_support': total_support_val
        }

        df = pd.DataFrame([data])[features]
        df_scaled = scaler.transform(df)
        probability = model.predict_proba(df_scaled)[0, 1]
        prediction = 1 if probability >= config['optimal_threshold'] else 0

        # Results
        st.divider()
        st.header("Prediction Results")

        col_a, col_b, col_c = st.columns([1, 2, 1])

        with col_b:
            if prediction == 1:
                st.error("ALERT : Dropout risk detected")
            else:
                st.success("No alert : Student profile is stable")

            st.metric("Dropout probability", f"{probability:.1%}")
            st.progress(probability)

            st.markdown("---")
            st.subheader("Interpretation")

            if prediction == 1:
                risk_level = (
                    "Critical" if probability > 0.7
                    else "High" if probability > 0.5
                    else "Moderate"
                )
                st.markdown(f"**Risk level** : {risk_level}")
                st.markdown("**Identified risk factors** :")

                if failures > 0:
                    st.markdown(
                        f"- {failures} past failure(s) — primary instability signal"
                    )
                if absences > 15:
                    st.markdown(f"- {absences} absences — progressive disengagement")
                if goout >= 4:
                    st.markdown(
                        f"- High social activity ({goout}/5) — reduced study availability"
                    )
                if higher == "No":
                    st.markdown("- No higher education goal — low academic motivation")
                if total_alcohol_val >= 6:
                    st.markdown(
                        f"- High alcohol consumption ({total_alcohol_val}/10)"
                    )

                st.markdown("""
**Recommended interventions**
1. Individual counseling session
2. Targeted academic tutoring
3. Attendance monitoring
4. Academic orientation support
                """)
            else:
                st.markdown("Stable profile. No urgent intervention required.")
                st.markdown("**Protective factors :**")
                if failures == 0:
                    st.markdown("- No past academic failure")
                if absences < 10:
                    st.markdown("- Good attendance record")
                if higher == "Yes":
                    st.markdown("- Clear higher education goal")
                if (Medu_val + Fedu_val) >= 6:
                    st.markdown("- Educated family environment")

with tab2:
    st.header("Model Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Variable Importance (SHAP)")
        st.bar_chart(pd.DataFrame({
            'Variable': [
                'failures', 'goout', 'total_support', 'age', 'reason'
            ],
            'Impact': [0.15, 0.12, 0.10, 0.08, 0.07]
        }).set_index('Variable'))
        st.markdown("""
**Top 3 predictors**
1. failures : Academic failure history
2. goout : Social outing frequency
3. total_support : Cumulative support index (engineered feature)
        """)

    with col2:
        st.subheader("Performance Metrics")
        st.markdown(f"""
| Metric    | Value  |
|-----------|--------|
| Recall    | **80.77%** |
| Precision | 44.68% |
| F1-Score  | 0.578  |
| AUC-ROC   | 0.734  |
| Threshold | {config['optimal_threshold']:.3f} |

**Design rationale**
The model maximizes recall at the cost of precision.
In a dropout prevention context, the cost of missing
an at-risk student exceeds the cost of a false alert.
        """)

with tab3:
    st.header("Documentation")
    st.markdown("""
## Dataset

- **Source** : Cortez & Silva (2008), UCI Machine Learning Repository
- **Size** : 395 students, 34 socio-behavioral variables
- **Task** : Binary classification (fail if final grade < 10/20)

## Methodology

**Grade exclusion strategy**
G1 and G2 (first and second term grades) were deliberately excluded
to enable prediction before any grades are available.
The model relies exclusively on entry-level socio-behavioral data.

## System Components

**Prediction interface**
Web application for educators to assess individual student risk profiles.

**Autonomous agent**
Automated weekly pipeline: literature monitoring, model evaluation,
retraining if performance degrades, and alert generation.

## Limitations

- Dataset limited to 395 Portuguese secondary school students (2008)
- 19% of dropout cases undetected (atypical profiles)
- 55% false alert rate (accepted trade-off)
- Requires revalidation before deployment in different contexts

## Reference

Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary
School Student Performance. Proceedings of 5th Annual Future Business
Technology Conference, Porto, Portugal.

## Source Code

[GitHub Repository](https://github.com/justinkelem708-ops/student-dropout-prediction)
    """)

st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Student Dropout Prediction System | Justin Kelem | Master AI/DS 2027"
    "</div>",
    unsafe_allow_html=True
)