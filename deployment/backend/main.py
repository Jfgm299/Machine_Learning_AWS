from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd 

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
model = joblib.load("./../../data/trained_models/housing/housing_pipeline_v4_xgboost.pkl")
print(model.feature_names_in_)

# Define the expected order and types for the DataFrame input
# NOTE: This list now strictly matches the user's final column list (excluding 'price').
PREDICT_COLUMNS = [
    'property_type',
    'old_new',
    'duration',
    'town_city',
    'district',
    'county',
    'sale_year',
    'sale_month'
]

CATEGORICAL_COLUMNS = [
    'property_type',
    'old_new',
    'duration',
    'town_city',
    'district',
    'county',
]


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
    sale_year: int
    sale_month: int  


# -------------------------------------------
# PREDICT ENDPOINT
# -------------------------------------------

@app.post("/housing/predict")
def predict(data: InputData):
    # We use a dictionary structure where the keys match the column names expected by the model.
    # sale_date is now included as requested by the final column schema.
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

    # CONVERT TO PANDAS DATAFRAME
    input_df = pd.DataFrame(input_data_dict)
    
    # 1. Re-order the columns to match the trained model's feature order
    input_df = input_df[PREDICT_COLUMNS]

    # 2. Explicitly cast known categorical columns to 'category' type
    for col in CATEGORICAL_COLUMNS:
        if col in input_df.columns:
            # Cast the type to 'category', which the pipeline's encoders prefer
            input_df[col] = input_df[col].astype('category')
        
    try:
        prediction = model.predict(input_df) # Pass the prepared DataFrame
        return {"prediction": float(prediction[0])}

    except Exception as e:
        # If prediction still crashes, print the DataFrame's info for better debugging
        print("🔥 Prediction crashed with input DataFrame info:")
        input_df.info()
        print("🔥 Prediction error:", e)
        # Return a better error structure
        return {"error": str(e)}


# -------------------------------------------
# HISTORY ENDPOINT (NEW)
# -------------------------------------------
@app.post("/housing/history")
def predict_history(data: InputData):
    # 1. Set start year to 1995 (typical start of UK housing records)
    start_year = 1995
    
    # 2. Set end year to the USER'S selected year
    # Ensure we at least show a small range if they pick a date before 1995
    end_year = max(start_year + 1, data.sale_year)
    
    years = list(range(start_year, end_year + 1))

    # 3. Create batch data
    batch_data = []
    for year in years:
        batch_data.append({
            'property_type': data.property_type,
            'old_new': data.old_new,
            'duration': data.duration,
            'town_city': data.town_city,
            'district': data.district,
            'county': data.county,
            'sale_year': year,
            'sale_month': data.sale_month 
        })

    # 4. Convert to DataFrame & Predict
    input_df = pd.DataFrame(batch_data)
    input_df = input_df[PREDICT_COLUMNS]

    for col in CATEGORICAL_COLUMNS:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')

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