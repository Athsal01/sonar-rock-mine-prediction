
import streamlit as st
import joblib
import pandas as pd


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Sonar AI Prediction",
    page_icon="🔊",
    layout="wide"
)


# --------------------------------
# 2. Custom CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #A78BFA;
    text-align: center;
    font-size: 38px;
}

h2, h3 {
    color: #C4B5FD;
}

p, label {
    color: #D1D5DB;
}

div.stButton > button {
    background-color: #7C3AED;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 30px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #6D28D9;
    color: white;
    border: 1px solid #A78BFA;
}

[data-testid="stMetric"] {
    background-color: #1E1B2E;
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# 3. Load Trained Model
# --------------------------------

model_bundle = joblib.load("sonar_model_bundle.pkl")

model = model_bundle["model"]
feature_names = model_bundle["feature_names"]


# --------------------------------
# 4. Dashboard Header
# --------------------------------

st.title("🔊 SONAR AI CLASSIFICATION")

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Underwater Object Detection System | Rock vs Mine"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

st.success("Machine Learning Model Loaded Successfully!")


# --------------------------------
# 5. Project Information
# --------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Model", "Logistic Regression")
col2.metric("Input Features", "60")
col3.metric("Classes", "2")

st.divider()


# --------------------------------
# 6. Sonar Measurements
# --------------------------------

st.header("📊 Sonar Measurements")

st.write(
    "Enter the 60 sonar signal measurements below. "
    "Each measurement must be between 0 and 1."
)

cols = st.columns(3)

values = []

for i in range(60):

    with cols[i % 3]:

        value = st.number_input(
            f"Feature {i + 1}",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.4f",
            key=f"feature_{i}"
        )

        values.append(value)


st.divider()


# --------------------------------
# 7. Prediction
# --------------------------------

st.header("🤖 Classification Result")

if st.button("PREDICT OBJECT"):

    if len(values) != 60:

        st.error("Exactly 60 measurements are required.")

    elif any(value < 0 or value > 1 for value in values):

        st.error("Measurements must be between 0 and 1.")

    else:

        # Convert input measurements into DataFrame

        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )

        # Predict object

        prediction = model.predict(input_df)

        # Calculate class probabilities

        probabilities = model.predict_proba(input_df)

        rock_probability = probabilities[0][
            list(model.classes_).index("R")
        ]

        mine_probability = probabilities[0][
            list(model.classes_).index("M")
        ]

        st.success("Input Validation Successful!")

        # Display classification result

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction[0] == "R":

                st.success("🪨 OBJECT IDENTIFIED: ROCK")

            else:

                st.error("💣 OBJECT IDENTIFIED: MINE")

        with result_col2:

            st.metric(
                "Predicted Class",
                "ROCK" if prediction[0] == "R" else "MINE"
            )


        # --------------------------------
        # 8. Prediction Probability
        # --------------------------------

        st.subheader("📈 Prediction Probability")

        confidence_col1, confidence_col2 = st.columns(2)

        with confidence_col1:

            st.write("🪨 Rock Probability")

            st.progress(float(rock_probability))

            st.metric(
                "Rock",
                f"{rock_probability * 100:.2f}%"
            )

        with confidence_col2:

            st.write("💣 Mine Probability")

            st.progress(float(mine_probability))

            st.metric(
                "Mine",
                f"{mine_probability * 100:.2f}%"
            )


# --------------------------------
# 9. Understanding Rock and Mine
# --------------------------------

st.divider()

st.header("📖 Understanding the Predictions")

rock_col, mine_col = st.columns(2)

with rock_col:

    st.subheader("🪨 Rock")

    st.write("""
    A rock is a natural underwater object.

    The model identifies sonar signal patterns
    that resemble those of rocks.
    """)

with mine_col:

    st.subheader("💣 Mine")

    st.write("""
    A mine is an underwater explosive object.

    The model identifies sonar signal patterns
    that resemble those of mines.
    """)

with st.expander("How does this Machine Learning model work?"):

    st.write("""
    1. The application receives 60 sonar measurements.
    2. The measurements are passed to the trained
       Logistic Regression model.
    3. The model predicts Rock or Mine.
    4. The application displays the predicted class
       and estimated class probabilities.
    """)


# --------------------------------
# 10. Model Performance Dashboard
# --------------------------------

st.divider()

st.header("📊 Model Performance")

st.write(
    "Evaluation results obtained from the test dataset "
    "(42 samples)."
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Accuracy", "80.95%")

metric2.metric("Precision (Mine)", "76.92%")

metric3.metric("Recall (Mine)", "90.91%")

metric4.metric("F1 Score (Mine)", "83.33%")


# --------------------------------
# 11. Confusion Matrix
# --------------------------------

st.subheader("Confusion Matrix")

st.write(
    "Rows represent actual classes and columns represent "
    "predicted classes."
)

confusion_data = pd.DataFrame(
    [[20, 2], [6, 14]],
    index=["Actual Mine", "Actual Rock"],
    columns=["Predicted Mine", "Predicted Rock"]
)

st.dataframe(
    confusion_data,
    width="stretch"
)

st.info(
    "These metrics are based on the 42-sample test set "
    "used during model evaluation."
)


# --------------------------------
# 12. About This Project
# --------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("""
    **Project:** Sonar Rock vs Mine Prediction

    **Algorithm:** Logistic Regression

    **Input:** 60 sonar signal measurements

    **Output:** Rock (R) or Mine (M)

    **Dataset:** UCI Sonar Dataset

    **Training Split:** 80%

    **Testing Split:** 20%

    **Framework:** Streamlit

    **Libraries:** Pandas, Scikit-learn, Joblib
    """)


# --------------------------------
# 13. Footer
# --------------------------------

st.divider()

st.markdown(
    "<p style='text-align:center; color:#A78BFA;'>"
    "SONAR AI | Machine Learning Project"
    "</p>",
    unsafe_allow_html=True
)
