
# 🔊 Sonar Rock vs Mine Prediction

A Machine Learning web application that classifies underwater sonar signals into two categories: Rock and Mine.

The application uses Logistic Regression to analyze 60 sonar signal measurements and predict the nature of an underwater object.

## 🚀 Project Overview

This project demonstrates how Machine Learning can be used for underwater object classification using sonar signal data.

Users can enter 60 sonar measurements through a Streamlit interface and receive:
- Predicted object classification
- Rock probability
- Mine probability
- Model performance information

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## 🧠 Machine Learning Algorithm

**Logistic Regression**

The model is trained using the UCI Sonar Dataset.

- Total input features: 60
- Output classes: Rock (R) and Mine (M)
- Training split: 80%
- Testing split: 20%

## 📊 Model Performance

Evaluation results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 80.95% |
| Precision (Mine) | 76.92% |
| Recall (Mine) | 90.91% |
| F1-score (Mine) | 83.33% |

### Confusion Matrix

| Actual / Predicted | Mine | Rock |
|---|---:|---:|
| Mine | 20 | 2 |
| Rock | 6 | 14 |

## 📁 Project Structure

```text
Sonar_Prediction/
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
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate into the project folder

```bash
cd Sonar_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv ml_env
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\ml_env\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🔍 How It Works

1. User enters 60 sonar measurements.
2. Input data is converted into a Pandas DataFrame.
3. The trained Logistic Regression model processes the input.
4. The model predicts Rock or Mine.
5. The application displays the prediction and class probabilities.

## 👨‍💻 Author

Machine Learning Project

---

⭐ If you find this project interesting, feel free to explore it!
