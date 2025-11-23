from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd

# PyCaret
from pycaret.regression import load_model, predict_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 🔵 ORIGINAL HOUSING SECTION (UNCHANGED)
# ============================================

@app.get("/")
def read_root():
    return {"message": "FastAPI Housing Model is running!"}


# Load housing model + encoder
model = joblib.load("./../../data/trained_models/housing/dtr_baseline_model.joblib")
print("properties")
print(model.feature_names_in_)

PREDICT_COLUMNS = [
    'sale_year',
    'property_type',
    'old_new',
    'duration',
    'town_city',
    'district',
    'county'
]

CATEGORICAL_COLUMNS = [
    'property_type',
    'old_new',
    'duration',
    'town_city',
    'district',
    'county'
]


class InputData(BaseModel):
    sale_date: str
    property_type: str
    old_new: str
    duration: str
    town_city: str
    district: str
    county: str
    sale_year: int 


@app.post("/housing/predict")
def predict(data: InputData):

    input_data_dict = {
        'sale_year': [data.sale_year],
        'property_type': [data.property_type],
        'old_new': [data.old_new],
        'duration': [data.duration],
        'town_city': [data.town_city],
        'district': [data.district],
        'county': [data.county]
    }

    input_df = pd.DataFrame(input_data_dict)

    input_df = input_df[PREDICT_COLUMNS]

    input_df[CATEGORICAL_COLUMNS] = encoder.transform(input_df[CATEGORICAL_COLUMNS])
        
    try:
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}

    except Exception as e:
        print("🔥 Prediction crashed with input DataFrame info:")
        input_df.info()
        print("🔥 Prediction error:", e)
        return {"error": str(e)}


@app.post("/housing/history")
def predict_history(data: InputData):
    start_year = 1995
    end_year = max(start_year + 1, data.sale_year)
    years = list(range(start_year, end_year + 1))

    batch_data = []
    for year in years:
        batch_data.append({
            'sale_year': year,
            'property_type': data.property_type,
            'old_new': data.old_new,
            'duration': data.duration,
            'town_city': data.town_city,
            'district': data.district,
            'county': data.county
        })

    input_df = pd.DataFrame(batch_data)
    input_df = input_df[PREDICT_COLUMNS]
    input_df[CATEGORICAL_COLUMNS] = encoder.transform(input_df[CATEGORICAL_COLUMNS])

    try:
        predictions = model.predict(input_df)
        
        history_result = []
        for i, year in enumerate(years):
            history_result.append({
                "year": year,
                "price": float(predictions[i])
            })
            
        return {"history": history_result}

    except Exception as e:
        print(f"🔥 History Prediction error: {e}")
        return {"error": str(e), "history": []}


# ======================================================
# 🟢 NEW ELECTRICITY PREDICTION ENDPOINT (ADDED HERE)
# ======================================================

print("⚡ Loading Electricity ND model...")
ELECTRICITY_MODEL_PATH = "./../../data/trained_models/electricity/ND_model"
electricity_model = load_model(ELECTRICITY_MODEL_PATH)
print("⚡ Electricity Model loaded successfully!")


class ElectricityInput(BaseModel):
    SETTLEMENT_DATE: str
    SETTLEMENT_PERIOD: int
    TSD: float

    EMBEDDED_WIND_GENERATION: float
    EMBEDDED_WIND_CAPACITY: float
    EMBEDDED_SOLAR_GENERATION: float
    EMBEDDED_SOLAR_CAPACITY: float

    NON_BM_STOR: float
    PUMP_STORAGE_PUMPING: float
    SCOTTISH_TRANSFER: float
    IFA_FLOW: float


@app.post("/electricity/predict")
def predict_electricity(data: ElectricityInput):

    date_obj = pd.to_datetime(data.SETTLEMENT_DATE)

    df = pd.DataFrame({
        'SETTLEMENT_DATE': [date_obj],
        'SETTLEMENT_PERIOD': [data.SETTLEMENT_PERIOD],
        'TSD': [data.TSD],
        'EMBEDDED_WIND_GENERATION': [data.EMBEDDED_WIND_GENERATION],
        'EMBEDDED_WIND_CAPACITY': [data.EMBEDDED_WIND_CAPACITY],
        'EMBEDDED_SOLAR_GENERATION': [data.EMBEDDED_SOLAR_GENERATION],
        'EMBEDDED_SOLAR_CAPACITY': [data.EMBEDDED_SOLAR_CAPACITY],
        'NON_BM_STOR': [data.NON_BM_STOR],
        'PUMP_STORAGE_PUMPING': [data.PUMP_STORAGE_PUMPING],
        'SCOTTISH_TRANSFER': [data.SCOTTISH_TRANSFER],
        'IFA_FLOW': [data.IFA_FLOW],
    })

    df['Year'] = df['SETTLEMENT_DATE'].dt.year
    df['Month'] = df['SETTLEMENT_DATE'].dt.month
    df['Day'] = df['SETTLEMENT_DATE'].dt.day
    df['Weekday'] = df['SETTLEMENT_DATE'].dt.weekday

    df_model = df.drop(columns=['SETTLEMENT_DATE'])

    prediction_df = predict_model(electricity_model, df_model)

    pred_value = float(prediction_df['prediction_label'].iloc[0])

    return {"prediction": pred_value}