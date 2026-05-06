import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="Customer Churn Prediction API")

# Load model relative to this file's location to work well on Vercel
# Vercel copies the api folder and root files.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class CustomerData(BaseModel):
    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    
    # Preprocess inputs
    # Gender: Female=0, Male=1
    gender_val = 1 if data.Gender == "Male" else 0
    
    # Geography: Germany or Spain (France is baseline: both 0)
    geo_germany = 1 if data.Geography == "Germany" else 0
    geo_spain = 1 if data.Geography == "Spain" else 0
    
    # Create DataFrame matching exact feature columns seen during training
    input_df = pd.DataFrame([{
        'CreditScore': data.CreditScore,
        'Gender': gender_val,
        'Age': data.Age,
        'Tenure': data.Tenure,
        'Balance': data.Balance,
        'NumOfProducts': data.NumOfProducts,
        'HasCrCard': data.HasCrCard,
        'IsActiveMember': data.IsActiveMember,
        'EstimatedSalary': data.EstimatedSalary,
        'Geography_Germany': geo_germany,
        'Geography_Spain': geo_spain
    }])
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    
    return {
        "churn_prediction": int(pred),
        "churn_probability": float(prob),
        "status": "Will Churn" if pred == 1 else "Will Stay"
    }
