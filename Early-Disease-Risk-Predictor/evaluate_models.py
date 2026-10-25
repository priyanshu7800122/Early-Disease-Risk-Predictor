
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    classification_report
)
from xgboost import XGBClassifier

# Load the original dataset
columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("diabetes.csv", header=None, names=columns)

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"]

# Treat suspicious zero measurements as missing
measurement_columns = [
    "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI"
]
X[measurement_columns] = X[measurement_columns].replace(0, float("nan"))

# Split BEFORE imputing missing values
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Positive-class Recall": round(recall, 3),
        "Positive-class Precision": round(precision, 3)
    })

    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Positive-class recall: {recall:.3f}")
    print(f"Positive-class precision: {precision:.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

pd.DataFrame(results).to_csv("model_comparison_clean.csv", index=False)

print("\nUpdated results saved to model_comparison_clean.csv")
