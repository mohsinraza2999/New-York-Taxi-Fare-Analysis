from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scripts import predict
import pandas as pd
from scripts import preprocessing, feature_engineering, train_model, evaluate, plot_utils

app = FastAPI(title="NYC Taxi Fare Prediction API")

app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def welcome_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# use this function to:
    # Preprocess, Clean, Feature Engineering
    # Training, Storing Results, Plots
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

@app.get("/form", response_class=HTMLResponse)
async def prediction_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/predict")
def predict_fare(request: Request,
    trip_duration: float = Form(...),
    trip_distance: float = Form(...),
    pickup_location_id: str = Form(...),
    dropoff_location_id: str = Form(...)):
    
    input_data={'duration': trip_duration,
    'trip_distance': trip_distance,
    'PULocationID': pickup_location_id,
    'DOLocationID': dropoff_location_id}
    result = predict.make_prediction(input_data)
    return HTMLResponse(content=f"<h2>Estimated Fare: ${result['predicted_fare']}</h2><a href='/form'>Try Again</a>", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)