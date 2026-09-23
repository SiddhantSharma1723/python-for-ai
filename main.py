from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd


app = FastAPI()

try:
    model = joblib.load("students_pass_model.joblib")
except Exception as e:
    raise RuntimeError(f"Could not load model {e}")

class PredictionInput(BaseModel):
    hours_studied: float = Field(ge=0)
    attendance_pct: float = Field(ge=0, le=100)
    prev_score: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model = PredictionResponse)
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{

        data.hours_studied,
        data.attendance_pct,
        data.prev_score,
        data.sleep_hours
    }])

    prediction= model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }