
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Early Disease Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Early Disease Risk Predictor")
st.write(
    "An educational machine-learning project that estimates "
    "diabetes risk from health measurements."
)

st.warning(
    "Educational use only. This model is not a medical device "
    "and must not be used to diagnose diabetes or guide treatment."
)

@st.cache_resource
def load_model():
    columns = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    df = pd.read_csv("diabetes.csv", header=None, names=columns)

    X = df.drop("Outcome", axis=1).copy()
    y = df["Outcome"]

    measurement_columns = [
        "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI"
    ]
    X[measurement_columns] = X[measurement_columns].replace(
        0, float("nan")
    )

    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    return model

model = load_model()

st.subheader("Enter health measurements")

with st.form("risk_form"):
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose level", min_value=1.0, max_value=300.0, value=110.0
    )
    blood_pressure = st.number_input(
        "Blood pressure (diastolic)", min_value=1.0,
        max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin thickness", min_value=1.0,
        max_value=100.0, value=20.0
    )
    insulin = st.number_input(
        "Insulin", min_value=1.0, max_value=900.0, value=80.0
    )
    bmi = st.number_input(
        "BMI", min_value=1.0, max_value=100.0, value=25.0
    )
    diabetes_pedigree = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input(
        "Age", min_value=21, max_value=100, value=30
    )

    submitted = st.form_submit_button("Estimate model score")

if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Model output")
    st.metric("Estimated model score", f"{probability * 100:.1f}%")
    st.progress(float(probability))

    st.caption(
        "This is the model's estimated probability for the positive "
        "class in its training dataset—not a validated personal "
        "medical risk estimate. Please consult a qualified healthcare "
        "professional for medical assessment."
    )
