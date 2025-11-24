import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib  # 👈 CORRECCIÓN 1: Usamos joblib para el modelo de Housing
from pycaret.regression import load_model, predict_model # Usamos PyCaret solo para Electricidad
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from calendar import monthrange
from typing import List, Dict

# Load environment
load_dotenv()
app = FastAPI()

# CORS config
origins_str = os.getenv("ALLOWED_ORIGINS", "*")
origins = origins_str.split(",") if origins_str != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔵 HOUSING SECTION
# =========================

# 1. Definimos las columnas EXACTAS del entrenamiento (Cell 4)
# NOTA: Eliminé 'sale_year' porque no estaba en tu Cell 4
PREDICT_COLUMNS = [
    'Year','Month','Day','Weekday',
    'property_type','old_new','duration',
    'town_city','district','county',
    'record_status___monthly_file_only'
]

categorical_features = [
    'property_type','old_new','duration',
    'town_city','district','county',
    'record_status___monthly_file_only'
]

# Load housing model
HOUSING_PATH = os.getenv("HOUSING_MODEL_PATH") # Asegúrate que apunte al .joblib o .pkl
if not os.path.exists(HOUSING_PATH):
    print(f"⚠️ Warning: Could not find housing model at {HOUSING_PATH}")

print(f"⏳ Loading housing model from {HOUSING_PATH}...")
try:
    # 👈 CORRECCIÓN: Usamos joblib porque tu entrenamiento fue con lgb.train directo
    housing_model = joblib.load(HOUSING_PATH) 
    print("✅ Housing model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load housing model: {e}")
    # Si falla joblib, intentamos pickle por si acaso
    try:
        import pickle
        with open(HOUSING_PATH, 'rb') as f:
            housing_model = pickle.load(f)
        print("✅ Housing model loaded with pickle!")
    except:
        raise e

# Input model
class InputData(BaseModel):
    date_of_transfer: str | None = None
    sale_date: str | None = None
    property_type: str
    old_new: str
    duration: str
    town_city: str
    district: str
    county: str
    sale_year: int

@app.get("/")
def read_root():
    env_name = os.getenv("APP_ENV", "unknown")
    return {"message": f"FastAPI Housing Model is running in {env_name} mode!"}

@app.post("/housing/predict")
def predict(data: InputData):
    date_str = data.date_of_transfer or data.sale_date
    if not date_str:
        return {"error": "No date provided"}

    date_obj = pd.to_datetime(date_str)

    # Creamos el DF
    input_df = pd.DataFrame([{
        'Year': date_obj.year,
        'Month': date_obj.month,
        'Day': date_obj.day,
        'Weekday': date_obj.weekday(),
        # 'sale_year': data.sale_year, 👈 ELIMINADO (no estaba en el entrenamiento)
        'property_type': data.property_type,
        'old_new': data.old_new,
        'duration': data.duration,
        'town_city': data.town_city,
        'district': data.district,
        'county': data.county,
        'record_status___monthly_file_only': 'A', 
    }])

    # Filtramos columnas
    input_df = input_df[PREDICT_COLUMNS]

    # 👈 CORRECCIÓN CRÍTICA: Convertir a category
    for col in categorical_features:
        input_df[col] = input_df[col].astype('category')

    try:
        prediction = housing_model.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        print("🔥 Prediction crashed:", e)
        # Debugging info
        print(input_df.dtypes)
        return {"error": str(e)}

@app.post("/housing/history")
def predict_history(data: InputData):
    date_str = data.date_of_transfer or data.sale_date
    if not date_str:
        return {"error": "No date provided"}

    start_year = 1995
    end_year = max(start_year + 1, data.sale_year)
    years = list(range(start_year, end_year + 1))

    batch_data = []
    for year in years:
        date_obj = pd.to_datetime(f"{year}-01-01")
        batch_data.append({
            'Year': date_obj.year,
            'Month': date_obj.month,
            'Day': date_obj.day,
            'Weekday': date_obj.weekday(),
            'property_type': data.property_type,
            'old_new': data.old_new,
            'duration': data.duration,
            'town_city': data.town_city,
            'district': data.district,
            'county': data.county,
            'record_status___monthly_file_only': 'A',
        })

    input_df = pd.DataFrame(batch_data)
    input_df = input_df[PREDICT_COLUMNS]

    # 👈 CORRECCIÓN CRÍTICA: Convertir a category en el loop histórico también
    for col in categorical_features:
        input_df[col] = input_df[col].astype('category')

    try:
        predictions = housing_model.predict(input_df)
        history_result = [{"year": year, "price": float(predictions[i])} for i, year in enumerate(years)]
        return {"history": history_result}
    except Exception as e:
        print(f"🔥 History Prediction error: {e}")
        return {"error": str(e), "history": []}

# =========================
# 🟢 ELECTRICITY SECTION
# =========================

ELEC_PATH = os.getenv("ELECTRICITY_MODEL_PATH")
if not ELEC_PATH:
    raise ValueError("❌ ELECTRICITY_MODEL_PATH no está definido en el .env")

# Aquí sí usamos load_model de PyCaret porque es un modelo distinto
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
    df = pd.DataFrame([{
        'SETTLEMENT_DATE': date_obj,
        'SETTLEMENT_PERIOD': data.SETTLEMENT_PERIOD,
        'TSD': data.TSD,
        'EMBEDDED_WIND_GENERATION': data.EMBEDDED_WIND_GENERATION,
        'EMBEDDED_WIND_CAPACITY': data.EMBEDDED_WIND_CAPACITY,
        'EMBEDDED_SOLAR_GENERATION': data.EMBEDDED_SOLAR_GENERATION,
        'EMBEDDED_SOLAR_CAPACITY': data.EMBEDDED_SOLAR_CAPACITY,
        'NON_BM_STOR': data.NON_BM_STOR,
        'PUMP_STORAGE_PUMPING': data.PUMP_STORAGE_PUMPING,
        'SCOTTISH_TRANSFER': data.SCOTTISH_TRANSFER,
        'IFA_FLOW': data.IFA_FLOW
    }])
    df['Year'] = df['SETTLEMENT_DATE'].dt.year
    df['Month'] = df['SETTLEMENT_DATE'].dt.month
    df['Day'] = df['SETTLEMENT_DATE'].dt.day
    df['Weekday'] = df['SETTLEMENT_DATE'].dt.weekday
    df_model = df.drop(columns=['SETTLEMENT_DATE'])
    
    # PyCaret se encarga de los tipos internamente, no tocamos nada aquí
    pred_df = predict_model(electricity_model, df_model)
    return {"prediction": float(pred_df['prediction_label'].iloc[0])}

@app.post("/electricity/monthly")
def electricity_monthly(data: ElectricityInput):
    try:
        date_obj = pd.to_datetime(data.SETTLEMENT_DATE)
        year, month = date_obj.year, date_obj.month
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
                    'IFA_FLOW': data.IFA_FLOW
                })
            df_day = pd.DataFrame(rows)
            df_day['Year'] = df_day['SETTLEMENT_DATE'].dt.year
            df_day['Month'] = df_day['SETTLEMENT_DATE'].dt.month
            df_day['Day'] = df_day['SETTLEMENT_DATE'].dt.day
            df_day['Weekday'] = df_day['SETTLEMENT_DATE'].dt.weekday
            df_day_model = df_day.drop(columns=['SETTLEMENT_DATE'])
            
            pred_df = predict_model(electricity_model, df_day_model)
            mean_val = float(pred_df['prediction_label'].mean())
            results.append({"date": f"{year}-{month:02d}-{day:02d}", "mean": mean_val})

        return {"monthly": results}
    except Exception as e:
        print("🔥 electricity_monthly error:", e)
        return {"error": str(e), "monthly": []}