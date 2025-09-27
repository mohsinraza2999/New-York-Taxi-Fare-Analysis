from fastapi import FastAPI
from scripts import predict
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from scripts import preprocessing, feature_engineering, train_model, evaluate, plot_utils

app = FastAPI(title="NYC Taxi Fare Prediction API")

@app.get("/")
def home():
    return {"message": "NYC Taxi Fare Prediction API is running."}

@app.get("/run")
def run_pipeline():
    try:
        df = pd.read_csv("data/raw/2017_Yellow_Taxi_Trip_Data.csv")
        df = preprocessing.load_and_clean_data(df)
        df = feature_engineering.add_features(df)
        model, scaler, X_test_scaled, y_test = train_model.train_model(df)
        results_df, mse, r2 = evaluate.evaluate_model(model, X_test_scaled, y_test)
        plot_utils.save_all_plots(df, y_test, model.predict(X_test_scaled), y_test.values - model.predict(X_test_scaled))

        return {
            "status": "✅ Pipeline executed successfully",
            "mse": round(mse, 2),
            "r2_score": round(r2, 3)
        }

    except Exception as e:
        return {"status": "❌ Error", "message": str(e)}


# -------------------------
# PREDICTION ENDPOINT
# -------------------------

class PredictionInput(BaseModel):
    tpep_pickup_datetime: str
    tpep_dropoff_datetime: str
    passenger_count: Optional[int] = 1
    trip_distance: float
    PULocationID: str
    DOLocationID: str
    payment_type: Optional[str] = None
    fare_amount: Optional[float] = None
    tip_amount: Optional[float] = None
    tolls_amount: Optional[float] = None
    total_amount: Optional[float] = None

@app.post("/predict")
def predict_fare(input_data: PredictionInput):
    input_dict = input_data.dict()
    result = predict.make_prediction(input_dict)
    return result

if __name__ == "_ _main__":
    import uvicorn
    uvicorn.run("main:main", host="0.0.0.0", port=8000, reload=True)