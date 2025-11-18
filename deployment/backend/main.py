from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# ROOT ROUTE
# -------------------------------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI Housing Model is running!"}

# -------------------------------------------
# LOAD MODEL
# -------------------------------------------
# IMPORTANT: This must point to your actual pipeline path
model = joblib.load("../../data/trained_models/housing/housing_pipeline_v4_xgboost.pkl")


# -------------------------------------------
# EXPECTED INPUT FROM REACT FRONTEND
# -------------------------------------------
class InputData(BaseModel):
    sale_date: str
    property_type: str
    old_new: str
    duration: str
    town_city: str
    district: str
    county: str
    record_status___monthly_file_only: str
    sale_year: int


# -------------------------------------------
# PREDICT ENDPOINT
# -------------------------------------------
from datetime import datetime

@app.post("/predict")
def predict(data: InputData):
    print(data.sale_date)
    # Ensure date is converted to YYYY-MM-DD
    try:
        # If frontend sends: "2025-11-18"
        
        sale_date_clean = datetime.fromisoformat(data.sale_date).strftime("%Y-%m-%d")
    except:
        try:
            # If frontend sends: "11/18/2025"
            sale_date_clean = datetime.strptime(data.sale_date, "%m/%d/%Y").strftime("%Y-%m-%d")
        except:
            print(data.sale_date)
            return {"error": "Invalid date format. Expected YYYY-MM-DD."}

    input_vector = [[
        sale_date_clean,
        data.property_type,
        data.old_new,
        data.duration,
        data.town_city,
        data.district,
        data.county,
        data.record_status___monthly_file_only,
        data.sale_year
    ]]

    try:
        prediction = model.predict(input_vector)
        return {"prediction": float(prediction[0])}

    except Exception as e:
        print("🔥 Prediction crashed:", e)
        return {"error": str(e)}