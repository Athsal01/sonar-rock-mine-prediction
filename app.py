
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Sonar Rock vs Mine Prediction",
    page_icon="🌊",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------

model_bundle = joblib.load("sonar_model_bundle.pkl")

model = model_bundle["model"]
scaler = model_bundle["scaler"]
feature_names = model_bundle["feature_names"]

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* Headings */
h1, h2, h3 {
    color: #c084fc !important;
}

/* General text */
p, li {
    color: #e2e8f0 !important;
}

/* Input labels */
.stNumberInput label,
.stTextArea label,
.stRadio label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Number input boxes */
.stNumberInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #a855f7 !important;
    border-radius: 8px !important;
}

/* Bulk text area */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #a855f7 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
}

/* Radio buttons */
.stRadio div[role="radiogroup"] label {
    color: #ffffff !important;
}

/* Buttons */
div.stButton > button {
    background-color: #9333ea !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #c084fc !important;
    padding: 10px 25px !important;
    font-weight: bold !important;
}

div.stButton > button:hover {
    background-color: #7e22ce !important;
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #9333ea;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #ffffff !important;
}

/* Captions */
.stCaption {
    color: #cbd5e1 !important;
}

/* Dividers */
hr {
    border-color: #475569;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.title("🌊 Sonar Rock vs Mine Prediction")

st.markdown("""
Predict whether an underwater sonar signal indicates
a **Rock 🪨** or a **Mine 💣** using Machine Learning.
""")

st.markdown("---")


# --------------------------------
# MODEL PERFORMANCE
# --------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "83.33%")
col2.metric("Precision", "77.78%")
col3.metric("Recall", "95.45%")
col4.metric("F1 Score", "85.71%")

st.markdown("---")


# --------------------------------
# INPUT SECTION
# --------------------------------

st.subheader("🔍 Enter Sonar Data")

input_method = st.radio(
    "Choose Input Method",
    ["Bulk Input (Paste 60 Values)", "Individual Inputs"],
    horizontal=True
)


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------

def make_prediction(values):

    input_df = pd.DataFrame(
        [values],
        columns=feature_names
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    mine_index = list(model.classes_).index("M")
    rock_index = list(model.classes_).index("R")

    mine_probability = probabilities[mine_index] * 100
    rock_probability = probabilities[rock_index] * 100

    return prediction, mine_probability, rock_probability


# --------------------------------
# DISPLAY RESULT FUNCTION
# --------------------------------

def display_result(prediction, mine_probability, rock_probability):

    st.markdown("---")

    st.subheader("🎯 Prediction Result")

    if prediction == "M":
        st.error("⚠️ Prediction: MINE")
    else:
        st.success("🪨 Prediction: ROCK")

    col1, col2 = st.columns(2)

    col1.metric(
        "Mine Probability",
        f"{mine_probability:.2f}%"
    )

    col2.metric(
        "Rock Probability",
        f"{rock_probability:.2f}%"
    )

    st.write("### Prediction Confidence")

    st.write("💣 Mine Probability")
    st.progress(float(mine_probability / 100))

    st.write("🪨 Rock Probability")
    st.progress(float(rock_probability / 100))


# --------------------------------
# BULK INPUT
# --------------------------------

if input_method == "Bulk Input (Paste 60 Values)":

    st.markdown("""
    <div style="
        background-color: #2e1065;
        color: #ffffff;
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #a855f7;
        margin-bottom: 15px;
        font-size: 16px;
        font-weight: 600;
    ">
        📋 Paste all 60 numerical values separated by commas.
    </div>
    """, unsafe_allow_html=True)

    bulk_input = st.text_area(
        "Sonar Feature Values",
        placeholder="0.0491, 0.0279, 0.0592, 0.1270, ...",
        height=160
    )

    if st.button("🚀 Predict", key="bulk_predict"):

        try:

            cleaned_input = bulk_input.replace("\n", ",")

            values = [
                float(value.strip())
                for value in cleaned_input.split(",")
                if value.strip()
            ]

            if len(values) != 60:

                st.error(
                    f"Expected 60 values, but received {len(values)}."
                )

            elif not np.isfinite(values).all():

                st.error("Please enter valid finite numbers.")

            elif not all(0 <= value <= 1 for value in values):

                st.error("Each feature value must be between 0 and 1.")

            else:

                prediction, mine_probability, rock_probability = (
                    make_prediction(values)
                )

                display_result(
                    prediction,
                    mine_probability,
                    rock_probability
                )

        except ValueError:

            st.error(
                "Invalid input! Please enter numerical values separated by commas."
            )


# --------------------------------
# INDIVIDUAL INPUT
# --------------------------------

elif input_method == "Individual Inputs":

    st.markdown("""
    <div style="
        background-color: #2e1065;
        color: #ffffff;
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #a855f7;
        margin-bottom: 15px;
        font-size: 16px;
        font-weight: 600;
    ">
        ✏️ Enter all 60 sonar feature values individually.
    </div>
    """, unsafe_allow_html=True)

    values = []

    cols = st.columns(3)

    for i in range(60):

        with cols[i % 3]:

            value = st.number_input(
                f"Feature {i + 1}",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.001,
                format="%.4f",
                key=f"feature_{i}"
            )

            values.append(value)

    if st.button("🚀 Predict", key="individual_predict"):

        prediction, mine_probability, rock_probability = (
            make_prediction(values)
        )

        display_result(
            prediction,
            mine_probability,
            rock_probability
        )


# --------------------------------
# ABOUT PROJECT
# --------------------------------

st.markdown("---")

st.subheader("ℹ️ About This Project")

st.markdown("""
This project uses **Logistic Regression** to classify underwater
sonar signals into two categories:

- 🪨 **Rock (R)**
- 💣 **Mine (M)**

### Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- StandardScaler
- Logistic Regression
- Streamlit

### Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Scaling using StandardScaler
4. Model Training
5. Model Evaluation
6. Model Deployment using Streamlit
""")

st.markdown("---")

st.caption(
    "Sonar Rock vs Mine Prediction | Machine Learning Project 🚀"
)
