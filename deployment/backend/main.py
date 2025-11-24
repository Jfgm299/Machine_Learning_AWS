import os
from dotenv import load_dotenv # 👈 Importante para leer el .env

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd

# PyCaret
from pycaret.regression import load_model, predict_model

# 1. Cargar variables de entorno al inicio
load_dotenv()

app = FastAPI()

# 2. Configuración de CORS dinámica
# Leemos la variable y la convertimos en una lista separando por comas
origins_str = os.getenv("ALLOWED_ORIGINS", "*")
origins = origins_str.split(",") if origins_str != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 👈 Usa la lista del .env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 🔵 HOUSING SECTION
# ============================================

@app.get("/")
def read_root():
    env_name = os.getenv("APP_ENV", "unknown")
    return {"message": f"FastAPI Housing Model is running in {env_name} mode!"}


# Load housing model + encoder
# 3. Usar rutas del .env
HOUSING_PATH = os.getenv("HOUSING_MODEL_PATH")

if not HOUSING_PATH or not os.path.exists(HOUSING_PATH):
    raise FileNotFoundError(f"❌ No se encontró el modelo de Housing en: {HOUSING_PATH}")

model = joblib.load(HOUSING_PATH)
print(f"✅ Housing model loaded from: {HOUSING_PATH}")
# print(model.feature_names_in_) 

PREDICT_COLUMNS = [
    'sale_year', 'property_type', 'old_new', 'duration', 
    'town_city', 'district', 'county'
]

CATEGORICAL_COLUMNS = [
    'property_type', 'old_new', 'duration', 
    'town_city', 'district', 'county'
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
    
    # Nota: Asumo que 'encoder' está definido globalmente o cargado con el modelo.
    # Si no lo tienes definido aquí, asegúrate de cargarlo igual que el modelo.
    # input_df[CATEGORICAL_COLUMNS] = encoder.transform(input_df[CATEGORICAL_COLUMNS])
        
    try:
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}

    except Exception as e:
        print("🔥 Prediction crashed:", e)
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
    # input_df[CATEGORICAL_COLUMNS] = encoder.transform(input_df[CATEGORICAL_COLUMNS])

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
# 🟢 ELECTRICITY PREDICTION ENDPOINT
# ======================================================

# 4. Cargar ruta del modelo eléctrico desde el .env
ELEC_PATH = os.getenv("ELECTRICITY_MODEL_PATH")
print(f"⚡ Loading Electricity ND model from {ELEC_PATH}...")

if not ELEC_PATH: # PyCaret load_model maneja su propia validación de archivo, pero verificamos que la variable exista
    raise ValueError("❌ ELECTRICITY_MODEL_PATH no está definido en el .env")

electricity_model = load_model(ELEC_PATH)
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


# -------------------------------------------
# BULK / MONTHLY PREDICTIONS ENDPOINT
# -------------------------------------------
from calendar import monthrange
from typing import List, Dict

@app.post("/electricity/monthly")
def electricity_monthly(data: ElectricityInput):
    try:
        date_obj = pd.to_datetime(data.SETTLEMENT_DATE)
        year = date_obj.year
        month = date_obj.month
        ndays = monthrange(year, month)[1] 

        results: List[Dict] = []

        for day in range(1, ndays + 1):
            rows = []
            for period in range(1, 49):
                rows.append({
                    'SETTLEMENT_DATE': pd.to_datetime(f"{year}-{month:02d}-{day:02d}"),
                    'SETTLEMENT_PERIOD': period,
                    'TSD': data.TSD,
                    'EMBEDDED_WIND_GENERATION': data.EMBEDDED_WIND_GENERATION,
                    'EMBEDDED_WIND_CAPACITY': data.EMBEDDED_WIND_CAPACITY,
                    'EMBEDDED_SOLAR_GENERATION': data.EMBEDDED_SOLAR_GENERATION,
                    'EMBEDDED_SOLAR_CAPACITY': data.EMBEDDED_SOLAR_CAPACITY,
                    'NON_BM_STOR': data.NON_BM_STOR,
                    'PUMP_STORAGE_PUMPING': data.PUMP_STORAGE_PUMPING,
                    'SCOTTISH_TRANSFER': data.SCOTTISH_TRANSFER,
                    'IFA_FLOW': data.IFA_FLOW,
                })

            df_day = pd.DataFrame(rows)
            df_day['Year'] = df_day['SETTLEMENT_DATE'].dt.year
            df_day['Month'] = df_day['SETTLEMENT_DATE'].dt.month
            df_day['Day'] = df_day['SETTLEMENT_DATE'].dt.day
            df_day['Weekday'] = df_day['SETTLEMENT_DATE'].dt.weekday

            df_day_model = df_day.drop(columns=['SETTLEMENT_DATE'])
            pred_df = predict_model(electricity_model, df_day_model)
            preds = pred_df['prediction_label'].values
            mean_val = float(preds.mean())

            results.append({
                "date": f"{year}-{month:02d}-{day:02d}",
                "mean": mean_val
            })

        return {"monthly": results}

    except Exception as e:
        print("🔥 electricity_monthly error:", e)
        return {"error": str(e), "monthly": []}