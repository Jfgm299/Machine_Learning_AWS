from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd 
import os # Importar 'os' para manejar rutas de archivos

app = FastAPI(
    title="ML Prediction API",
    version="1.0.0",
    description="API para modelos de predicción de Vivienda y Electricidad."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# GLOBAL CONFIGURATION
# -------------------------------------------
# Rutas de los modelos
HOUSING_MODEL_PATH = "./../../data/trained_models/housing/housing_pipeline_v4_xgboost.pkl"
ELECTRICITY_MODEL_PATH = "./../../data/trained_models/electricity/electricity_pipeline.pkl" # Ruta placeholder
MODELS = {} # Diccionario para almacenar los modelos cargados

# Columnas esperadas por el modelo de VIVIENDA
HOUSING_PREDICT_COLUMNS = [
    'property_type',
    'old_new',
    'duration',
    'town_city',
    'district',
    'county',
    'sale_year',
    'sale_month'
]

HOUSING_CATEGORICAL_COLUMNS = [
    'property_type',
    'old_new',
    'duration',
    'town_city',
    'district',
    'county',
]

# -------------------------------------------
# LOAD MODELS (Se ejecuta al iniciar la API)
# -------------------------------------------
def load_model(path: str, model_name: str):
    """Carga un modelo de machine learning si existe."""
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            MODELS[model_name] = model
            print(f"✅ Modelo '{model_name}' cargado desde: {path}")
            # print(f"  -> Features esperadas: {model.feature_names_in_}")
        except Exception as e:
            print(f"❌ Error al cargar el modelo '{model_name}' desde {path}: {e}")
    else:
        print(f"⚠️ Advertencia: El archivo del modelo '{model_name}' no se encontró en: {path}")

# Cargar los modelos al inicio
load_model(HOUSING_MODEL_PATH, "housing")
# load_model(ELECTRICITY_MODEL_PATH, "electricity") # Puedes descomentar esto cuando el modelo esté listo

# -------------------------------------------
# ROOT ROUTE
# -------------------------------------------
@app.get("/")
def read_root():
    """Ruta raíz de la API."""
    loaded_models = ", ".join(MODELS.keys()) if MODELS else "Ninguno"
    return {"message": "FastAPI ML Prediction API está corriendo.", "modelos_cargados": loaded_models}


# -------------------------------------------
# INPUT DATA MODELS (Schemas Pydantic)
# -------------------------------------------
# Esquema de entrada para el modelo de VIVIENDA
class InputDataHousing(BaseModel):
    sale_date: str
    property_type: str
    old_new: str
    duration: str
    town_city: str
    district: str
    county: str
    sale_year: int
    sale_month: int  

# Esquema de entrada para el modelo de ELECTRICIDAD (PLACEHOLDER)
class InputDataElectricity(BaseModel):
    # Definir los campos necesarios para tu modelo de electricidad cuando lo tengas
    date: str
    temperature: float
    is_weekend: bool
    # ... otros campos ...


# -------------------------------------------
# HOUSING ENDPOINTS
# -------------------------------------------

@app.post("/housing/predict")
def predict_housing(data: InputDataHousing):
    """Realiza una predicción del precio de la vivienda."""
    
    if "housing" not in MODELS:
        raise HTTPException(status_code=503, detail="Modelo de Vivienda no cargado.")

    model = MODELS["housing"]

    # 1. Preparar los datos de entrada
    input_data_dict = {
        'property_type': [data.property_type],
        'old_new': [data.old_new],
        'duration': [data.duration],
        'town_city': [data.town_city],
        'district': [data.district],
        'county': [data.county],
        'sale_year': [data.sale_year],
        'sale_month': [data.sale_month]
    }

    input_df = pd.DataFrame(input_data_dict)
    
    # 2. Re-ordenar y castear tipos
    try:
        input_df = input_df[HOUSING_PREDICT_COLUMNS]
    except KeyError as e:
        # Esto ocurre si el frontend envía un campo incorrecto o faltante
        raise HTTPException(status_code=400, detail=f"Falta una columna necesaria: {e}. Columnas esperadas: {HOUSING_PREDICT_COLUMNS}")


    for col in HOUSING_CATEGORICAL_COLUMNS:
        if col in input_df.columns:
            # Castear a 'category'
            input_df[col] = input_df[col].astype('category')
        
    try:
        # 3. Predecir
        prediction = model.predict(input_df)
        # Devolver el valor de la predicción
        return {"prediction": float(prediction[0])}

    except Exception as e:
        print("🔥 La predicción de Vivienda falló con error:", e)
        # Devolver un error HTTP si la predicción falla internamente
        raise HTTPException(status_code=500, detail=f"Error interno del modelo de predicción: {e}")

# -------------------------------------------
# ELECTRICITY ENDPOINTS (PLACEHOLDER)
# -------------------------------------------

@app.post("/electricity/predict")
def predict_electricity(data: InputDataElectricity):
    """Realiza una predicción del consumo de electricidad."""
    
    # Aquí irá la lógica para cargar y usar el modelo de electricidad.
    if "electricity" not in MODELS:
        # Simplemente devolvemos un mensaje de error 503 (Servicio no disponible)
        # hasta que el modelo esté implementado.
        raise HTTPException(status_code=503, detail="Modelo de Electricidad no implementado o no cargado.")
        
    # Lógica futura:
    # model = MODELS["electricity"]
    # input_df = pd.DataFrame({...})
    # prediction = model.predict(input_df)
    # return {"prediction": float(prediction[0])}