
import pandas as pd

results = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy (%)": [73.38, 74.03, 76.62],
    "Positive-class Recall": [0.704, 0.685, 0.630],
    "Positive-class Precision": [0.603, 0.617, 0.680]
}

df = pd.DataFrame(results)

df.to_csv("model_comparison.csv", index=False)

print(df.to_string(index=False))
print("\nResults saved to model_comparison.csv")
