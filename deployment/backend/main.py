from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# Load your model
model = joblib.load("../../data/trained_models/housing/housing_pipeline_v4_xgboost.pkl")

class InputData(BaseModel):
    feature: float

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.feature]])
    return {"prediction": float(prediction[0])}