
print("Step1: Importing the Libraries")
print()

import gradio as gr
import pandas as pd
import joblib
from pathlib import Path


print("Step2: Loading the Model")
print()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE_PATH = BASE_DIR / "data" / "raw" / "airline_flight.csv"

MODEL_PATH = BASE_DIR / "models" / "airline_model.pkl"

model = joblib.load(MODEL_PATH)

df = pd.read_csv(RAW_FILE_PATH)

print("Model Loaded Successfully!")


print("Step3: Creating Prediction Function")
print()

def predict_flight(departure_hour, distance, weather, traffic):

    if distance is None or distance <= 0:
        return "Please enter a valid flight distance!"

    if departure_hour is None or not 0 <= departure_hour <= 23:
        return "Please enter a valid departure hour!"

    if weather is None or traffic is None:
        return "Please select weather and traffic!"

    input_data = pd.DataFrame([{
        "departure_hour": departure_hour,
        "distance": distance,
        "weather": weather,
        "traffic": traffic
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        status = "DELAYED"
        advice = "Please check airline notifications and contact customer support."
    else:
        status = "ON TIME"
        advice = "Please arrive at the airport on time."

    result = (
        f"Flight Status: {status}\n\n"
        f"Delay Probability: {probability * 100:.2f}%\n\n"
        f"Customer Advice: {advice}"
    )

    print(result)  # Displays the result in Command Prompt

    return result


print("Step4: Creating Gradio Interface")
print()


app = gr.Interface(

    fn=predict_flight,

    inputs=[

        gr.Slider(
            minimum=0,
            maximum=23,
            value=18,
            step=1,
            label="Departure Hour"
        ),

        gr.Number(
            value=1500,
            label="Flight Distance (KM)"
        ),

        gr.Dropdown(
            choices=[
                ("Good", 0),
                ("Moderate", 1),
                ("Bad", 2)
            ],
            value=2,
            label="Weather Condition"
        ),

        gr.Dropdown(
            choices=[
                ("Low", 0),
                ("Medium", 1),
                ("High", 2)
            ],
            value=2,
            label="Airport Traffic"
        )

    ],

    outputs=gr.Textbox(
    label="Flight Prediction Result",
    lines=5
),

       title="Airline Flight Delay Prediction System",


    description="Enter flight details to predict flight delay status.",

    flagging_mode="never"
)

app.launch()