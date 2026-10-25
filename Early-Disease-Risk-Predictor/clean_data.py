
import pandas as pd

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv("diabetes.csv", header=None, names=columns)

# These zero measurements may represent missing data
columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zero measurements with missing values
df[columns_to_clean] = df[columns_to_clean].replace(0, float("nan"))

print("Missing values before filling:")
print(df[columns_to_clean].isnull().sum())

# Fill missing measurements with their median
for column in columns_to_clean:
    df[column] = df[column].fillna(df[column].median())

# Save the cleaned dataset
df.to_csv("diabetes_cleaned.csv", index=False)

print("\nMissing values after filling:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)
print("\nCleaned dataset saved as diabetes_cleaned.csv")
