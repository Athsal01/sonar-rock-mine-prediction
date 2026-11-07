
# 🔊 Sonar Rock vs Mine Prediction

An end-to-end Machine Learning project that classifies underwater objects as **Rock (R)** or **Mine (M)** using sonar signal data.

Built using Python, Scikit-learn, and Streamlit.

## 🚀 Project Overview

This project uses sonar signal measurements to predict whether an underwater object is a rock or a mine.

A Logistic Regression classification model is trained using the UCI Sonar dataset. The trained model is integrated into an interactive Streamlit web application.

## ✨ Features

- Predicts Rock or Mine using 60 sonar input features.
- Interactive Streamlit web interface.
- Displays prediction confidence.
- Shows Rock and Mine probabilities.
- Model performance dashboard.
- Confusion matrix visualization.
- Easy-to-use prediction interface.

## 📸 Application Screenshots

### 1. Main Dashboard

![Sonar Dashboard](screenshots/Sonar_dashboard01.png.png)

### 2. Prediction Result

![Prediction Result](screenshots/Sonar_dashboard02.png.png)

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- NumPy
- Joblib
- Git and GitHub

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

**Dataset:** UCI Sonar Dataset

**Input Features:** 60 sonar signal measurements

**Output Classes:**
- R – Rock
- M – Mine

## 📊 Model Performance

The model was evaluated using a test dataset containing 42 samples.

| Metric | Score |
|---|---:|
| Accuracy | 80.95% |
| Precision (Mine) | 76.92% |
| Recall (Mine) | 90.91% |
| F1 Score (Mine) | 83.33% |

### Confusion Matrix

```text
[[20  2]
 [ 6 14]]
```

The matrix rows represent actual classes (Mine, Rock), and columns represent predicted classes (Mine, Rock).

These results represent the particular test split used during evaluation and do not guarantee real-world performance.

## 📂 Project Structure

```text
Sonar_Prediction/
│
├── screenshots/
│   ├── sonar-dashboard.png
│   └── prediction-result.png
│
├── app.py
├── sonar_model_bundle.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Athsal01/sonar-rock-mine-prediction.git
```

### 2. Navigate to the project folder

```bash
cd sonar-rock-mine-prediction
```

### 3. Create a virtual environment

```bash
python -m venv ml_env
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\ml_env\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🔄 Project Workflow

1. Load the UCI Sonar dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Train a Logistic Regression model.
5. Evaluate model performance.
6. Save the trained model using Joblib.
7. Integrate the model into Streamlit.
8. Generate predictions through the web interface.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Machine Learning classification.
- Data handling using Pandas.
- Model training and evaluation.
- Prediction probability analysis.
- Model serialization using Joblib.
- Building interactive applications using Streamlit.
- Version control using Git and GitHub.

## 👨‍💻 Author

**Athsal**

B.Tech – Artificial Intelligence

GitHub: https://github.com/Athsal01

---

⭐ If you find this project interesting, feel free to explore the repository!
