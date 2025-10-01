import joblib
import pandas as pd
import numpy as np
from scripts.preprocessing import load_and_clean_data
from scripts.feature_engineering import add_features

MODEL_PATH = "model/reg_model.pkl"
SCALER_PATH = "model/scaler.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def make_prediction(input_data: dict):
    model, scaler = load_model()

    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # Generate pickup_dropoff
    #df_clean=load_and_clean_data(df)

    # Estimate duration
    df_f=add_features(df)

    # Prepare features for prediction
    X = df_f[['trip_distance', 'mean_distance', 'duration']]  # Change based on your trained model

    X_scaled = scaler.transform(X)
    predicted_fare = model.predict(X_scaled)[0]

    return {
        "predicted_fare": round(float(predicted_fare), 2)
    }