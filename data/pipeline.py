import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from pycaret.regression import setup, load_model, finalize_model, save_model, compare_models
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y VARIABLES
# =========================================================================

# Rutas de Modelos Existentes (Para cargar y reentrenar)
HOUSING_MODEL_PATH = "./../../data/trained_models/housing/lightgbm_housing_model.pkl"
ELECTRICITY_MODEL_PATH = "./../../data/trained_models/electricity/ND_model"

# Rutas de los Nuevos Datos (Asegúrate de que estas rutas sean correctas)
NEW_FILE_HOUSING = "route/to/the/new/housing/data.parquet"
NEW_FILE_ELECTRICITY = "route/to/the/new/electricity/data.parquet"

# Rutas para guardar los modelos reentrenados
OUTPUT_MODEL_HOUSING = "./../../data/trained_models/housing/lightgbm_housing_model_v2.pkl"
OUTPUT_MODEL_ELECTRICITY = "./../../data/trained_models/electricity/ND_model_v2"

# DIRECTORIO TEMPORAL PARA PROCESAMIENTO
PROCESSED_DATA_DIR = "./temp_processed_data/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# =========================================================================
# 🔵 SECCIÓN HOUSING (LightGBM)
# =========================================================================

# --- 1. HIPERPARÁMETROS LIGHTGBM ---
# Asumo que estos son los parámetros que usaste. AJUSTA SI ES NECESARIO.
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

def clean_and_transform_housing(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la lógica de limpieza y Feature Engineering del notebook."""
    
    # 2.A. Standardize Column Names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

    # 2.B. Filter Date Range & Convert Date
    df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'])
    # NOTA: Mantener solo 1995-2017 aquí NO permitiría reentrenar con datos nuevos de 2018+. 
    # Para reentrenar con nuevos datos, eliminaremos el filtro de fecha (o lo ajustaremos).
    # df = df[(df['date_of_transfer'].dt.year >= 1995) & (df['date_of_transfer'].dt.year <= 2017)]

    # 2.C. Handling Missing Values
    df = df.dropna(subset=['price'])
    cols_to_fill = ['town_city', 'district', 'county']
    for col in cols_to_fill:
        if col in df.columns:
            # Añadir 'Unknown' si es necesario (manejo robusto para categóricas)
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                 if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('Unknown')
            df[col] = df[col].fillna('Unknown')
            # Asegurar que sean categóricas para LightGBM
            df[col] = df[col].astype('category')


    # 2.D. Outlier Removal (Hard Cap)
    initial_count = len(df)
    df = df[(df['price'] > 100) & (df['price'] < 10_000_000)]
    print(f"Dropped {initial_count - len(df)} outliers.")

    # 3. FEATURE ENGINEERING
    df['year'] = df['date_of_transfer'].dt.year
    df['month'] = df['date_of_transfer'].dt.month
    df['day'] = df['date_of_transfer'].dt.day # Añadido para consistencia con tu API
    df['weekday'] = df['date_of_transfer'].dt.weekday # Añadido para consistencia con tu API


    # 4. DATA TYPES OPTIMIZATION
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    return df

def retrain_housing_model(new_data_path: str, model_path: str, output_path: str, params: dict):
    """Carga datos nuevos, los limpia, reentrena LightGBM y guarda la nueva versión."""
    print("\n--- Starting Housing Model Retraining ---")
    
    # 1. Cargar datos nuevos
    try:
        df_new = pd.read_parquet(new_data_path)
    except FileNotFoundError:
        print(f"❌ ERROR: New Housing data not found at {new_data_path}.")
        return

    # 2. Limpieza y Transformación
    df_processed = clean_and_transform_housing(df_new)
    
    X = df_processed.drop(columns=['price', 'date_of_transfer'])
    y = df_processed['price']

    # 3. Cargar el modelo existente (Transfer Learning)
    # NOTA: En LightGBM, esto se hace cargando el modelo y usando .fit(init_model=...)
    try:
        # joblib.load se usa para LightGBM.
        existing_model = joblib.load(model_path)
        print(f"✅ Existing Housing model loaded from {model_path}.")
    except Exception as e:
        print(f"⚠️ Could not load existing model, training from scratch: {e}")
        existing_model = None

    # 4. Reentrenamiento (Transfer Learning)
    # Se utiliza el modelo existente como punto de partida (init_model).
    lgbm_model = lgb.LGBMRegressor(**params)
    
    print("⏳ Starting LightGBM Re-training...")
    lgbm_model.fit(
        X, y,
        init_model=existing_model, # Usar el modelo existente como punto de partida
        categorical_feature='auto',
        eval_metric='rmse',
    )
    print("✅ LightGBM Re-training complete.")

    # 5. Evaluación (Opcional, pero recomendado)
    preds = lgbm_model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    print(f"📈 Evaluation on new data (RMSE): {rmse:.2f}")

    # 6. Guardar el nuevo modelo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(lgbm_model, output_path)
    print(f"💾 New Housing model saved to {output_path}")

# =========================================================================
# 🟢 SECCIÓN ELECTRICITY (PyCaret)
# =========================================================================

def clean_and_transform_electricity(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la lógica de limpieza y Feature Engineering del notebook para Electricity."""
    
    # Asegurar que la fecha sea datetime
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
    
    # Core Cleaning & Type Fixes
    df['Year'] = df['SETTLEMENT_DATE'].dt.year
    df['Year'] = df['Year'].astype('int16')

    # Fill NaNs
    df['TSD'].fillna(0, inplace=True)
    df['EMBEDDED_WIND_GENERATION'].fillna(0, inplace=True)
    df['EMBEDDED_WIND_CAPACITY'].fillna(0, inplace=True)
    df['EMBEDDED_SOLAR_GENERATION'].fillna(0, inplace=True)
    df['EMBEDDED_SOLAR_CAPACITY'].fillna(0, inplace=True)
    df['SCOTTISH_TRANSFER'].fillna(0, inplace=True)
    
    # Se asume que columnas adicionales (IFA2_FLOW, etc.) NO están en el modelo final, 
    # pero deben limpiarse si están presentes en los nuevos datos
    cols_to_fillna = ['IFA2_FLOW', 'BRITNED_FLOW', 'MOYLE_FLOW', 'EAST_WEST_FLOW', 'NEMO_FLOW', 'NSL_FLOW', 'ELECLINK_FLOW', 'VIKING_FLOW', 'GREENLINK_FLOW']
    for col in cols_to_fillna:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # Conversión y Transformación de Target
    df['ND'] = pd.to_numeric(df['ND'], errors='coerce')
    df['ND_log'] = np.log(df['ND'] + 1)
    
    return df

# --- 2. CONFIGURACIÓN DEL SETUP DE PYCARET ---
# Este es el setup EXACTO que usaste en tu notebook
def setup_pycaret(df_model: pd.DataFrame):
    """Ejecuta el setup de PyCaret para aplicar las transformaciones originales."""
    
    # 🚨 Es vital que este setup sea idéntico al original para la coherencia del pipeline.
    reg = setup(
        data=df_model,
        target='ND',
        session_id=123,
        numeric_features=['TSD','EMBEDDED_WIND_GENERATION','EMBEDDED_WIND_CAPACITY',
                        'EMBEDDED_SOLAR_GENERATION','EMBEDDED_SOLAR_CAPACITY','NON_BM_STOR','PUMP_STORAGE_PUMPING',
                        'SCOTTISH_TRANSFER','IFA_FLOW','Year','Month','Day','Weekday'],
        fold_strategy='timeseries',
        fold=5,
        transform_target=True, # PyCaret maneja la transformación logarítmica para 'ND_log'
        data_split_shuffle=False, 
        fold_shuffle=False       
    )
    return reg

def retrain_electricity_model(new_data_path: str, model_path: str, output_path: str):
    """Carga datos nuevos, reentrena el modelo PyCaret y guarda la nueva versión."""
    print("\n--- Starting Electricity Model Retraining ---")
    
    # 1. Cargar datos nuevos
    try:
        df_new = pd.read_parquet(new_data_path)
    except FileNotFoundError:
        print(f"❌ ERROR: New Electricity data not found at {new_data_path}.")
        return

    # 2. Limpieza y Transformación (hasta antes de generar features de fecha)
    df_processed = clean_and_transform_electricity(df_new)
    
    # 3. Feature Engineering (Creación de características temporales)
    # NOTA: Esto se hace antes del setup de PyCaret
    df_processed['Year'] = df_processed['SETTLEMENT_DATE'].dt.year
    df_processed['Month'] = df_processed['SETTLEMENT_DATE'].dt.month
    df_processed['Day'] = df_processed['SETTLEMENT_DATE'].dt.day
    df_processed['Weekday'] = df_processed['SETTLEMENT_DATE'].dt.weekday 
    
    # Seleccionar solo las columnas usadas en el modelo original (antes del setup)
    columns_to_use = [
        'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND', 'TSD',
        'EMBEDDED_WIND_GENERATION','EMBEDDED_WIND_CAPACITY','EMBEDDED_SOLAR_GENERATION',
        'EMBEDDED_SOLAR_CAPACITY','NON_BM_STOR','PUMP_STORAGE_PUMPING','SCOTTISH_TRANSFER','IFA_FLOW'
    ]
    df_model_input = df_processed[columns_to_use].drop(columns=['SETTLEMENT_DATE'])

    # 4. Configurar PyCaret y Cargar Modelo
    # Ejecutar setup para aplicar transformaciones (escalado, etc.) a los nuevos datos
    setup_pycaret(df_model_input)
    
    # Cargar el modelo existente (el pipeline)
    try:
        # PyCaret carga el pipeline de preprocesamiento y el modelo final.
        existing_pipeline = load_model(model_path)
        print(f"✅ Existing PyCaret pipeline loaded from {model_path}.")
    except Exception as e:
        print(f"❌ ERROR: Could not load PyCaret model: {e}")
        return

    # 5. Reentrenamiento / Finalización
    # Usamos finalize_model() para reentrenar el modelo con el dataset completo (train+test)
    # y los nuevos datos, manteniendo el tipo de modelo (best_model[0] de tu notebook)
    
    # Reentrenar sobre todos los datos (el modelo en el pipeline existente)
    print("⏳ Starting PyCaret Re-training/Finalization...")
    new_pipeline = finalize_model(existing_pipeline)
    
    print("✅ PyCaret Re-training complete.")

    # 6. Guardar el nuevo modelo
    # PyCaret requiere guardar el nombre del archivo sin extensión (.pkl)
    save_model(new_pipeline, output_path) 
    print(f"💾 New Electricity model saved to {output_path}.pkl")

# =========================================================================
# 🚀 EJECUCIÓN DEL PIPELINE
# =========================================================================

if __name__ == '__main__':
    
    print(f"--- Starting Pipeline at {datetime.now()} ---")
    
    # 1. Reentrenar Modelo de Housing (LightGBM)
    # Asegúrate de que NEW_FILE_HOUSING apunta a tus datos nuevos
    retrain_housing_model(
        new_data_path=NEW_FILE_HOUSING,
        model_path=HOUSING_MODEL_PATH,
        output_path=OUTPUT_MODEL_HOUSING,
        params=LGBM_PARAMS
    )
    
    # 2. Reentrenar Modelo de Electricity (PyCaret)
    # Asegúrate de que NEW_FILE_ELECTRICITY apunta a tus datos nuevos
    retrain_electricity_model(
        new_data_path=NEW_FILE_ELECTRICITY,
        model_path=ELECTRICITY_MODEL_PATH,
        output_path=OUTPUT_MODEL_ELECTRICITY
    )
    
    print(f"\n--- Pipeline Complete at {datetime.now()} ---")