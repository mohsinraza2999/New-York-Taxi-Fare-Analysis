from fastapi import FastAPI
import pandas as pd
from scripts import preprocessing, feature_engineering, train_model, evaluate_model, plot_utils

app = FastAPI(title="NYC Taxi Fare Prediction API")

@app.get("/")
def read_root():
    return {"message": "NYC Taxi Fare Prediction API is running."}

@app.get("/run")
def run_pipeline():
    try:
        # Load raw data
        df = pd.read_csv("data/raw/your_file.csv")  # Replace with actual file path

        # Clean and preprocess
        df = preprocessing.load_and_clean_data(df)
        df = feature_engineering.add_features(df)

        # Train model
        model, scaler, X_test_scaled, y_test = train_model.train_model(df)

        # Evaluate
        results_df, mse, r2 = evaluate_model.evaluate_model(model, X_test_scaled, y_test)

        # Save plots
        plot_utils.save_all_plots(df, y_test, model.predict(X_test_scaled), y_test.values - model.predict(X_test_scaled))

        return {
            "status": "✅ Pipeline executed successfully",
            "mse": round(mse, 2),
            "r2_score": round(r2, 3),
            "saved_files": {
                "metrics": "outputs/reports/metrics.json",
                "figures": "outputs/figures/",
                "model": "models/reg_model.pkl"
            }
        }

    except Exception as e:
        return {"status": "❌ Error", "message": str(e)}