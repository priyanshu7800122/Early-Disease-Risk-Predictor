
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("diabetes_cleaned.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    ),

    "XGBoost": make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )
    )
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 30)
    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2), "%"
    )
    print(
        "Positive-class recall:",
        round(recall_score(y_test, predictions), 3)
    )
    print(
        "Positive-class precision:",
        round(precision_score(y_test, predictions), 3)
    )
