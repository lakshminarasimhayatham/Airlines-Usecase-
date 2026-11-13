
print("Step1: Importing the Libraries")
print()

import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Step2: Loading the Dataset")
print()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE_PATH = BASE_DIR / "data" / "raw" / "airline_flight.csv"
MODEL_PATH = BASE_DIR / "models" / "airline_model.pkl"

df = pd.read_csv(RAW_FILE_PATH)

print(df.head())


print("Step3: Splitting the Data: Opt")
print()


print("Step4: Data Preparation")
print()

X = df[["departure_hour", "distance", "weather", "traffic"]]
y = df["delay"]


print("Step5: Model Creation")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("Step6: Model Training")
print()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



print("Step7: Prediction")
print()

y_pred = model.predict(X_test)
print("Predictions:", y_pred)


print("Step8: Model Evaluation")
print()

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))


print("Step9: Saving the Model")
print()

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)



print("Step10: Testing New Flight")
print()

new_flight = pd.DataFrame([{
    "departure_hour": 18,
    "distance": 2000,
    "weather": 2,
    "traffic": 2
}])

prediction = model.predict(new_flight)

if prediction[0] == 1:
    print("Flight Status: Delayed")
else:
    print("Flight Status: On Time")


