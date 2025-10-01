# New-York-Taxi-Fare-Analysis
Analyzed NYC taxi fare data for TLC to build a fare estimation app. Conducted EDA, hypothesis testing, and regression analysis to identify key variables. Developed ML models (Random Forest, XGBoost) with GridSearchCV for hyperparameter tuning to predict fares accurately.
# NYC Taxi Fare Prediction 🚖

An end‑to‑end machine learning project that analyzes and models taxi fare data from New York City, provided by the NYC Taxi and Limousine Commission (TLC). The goal is to build a full‑stack fare estimation app that allows users to input trip details and receive an estimated fare in real time.

## 📌 Project Objective

In this fictional scenario, the NYC TLC has tasked the Automatidata team with building an app to help estimate taxi fares before a ride. This project covers the entire ML lifecycle:
- Data exploration and preprocessing
- Hypothesis testing and regression analysis
- Feature engineering
- Model training and hyperparameter tuning
- Backend API with FastAPI
- Frontend web interface
- Containerization with Docker
- CI/CD pipeline for automated testing and deployment
- Unit testing with pytest


## 🧪 Methodology

- Data Understanding – Reviewing raw TLC trip data
- Exploratory Data Analysis (EDA) – Identifying trends, anomalies, and distributions
- Hypothesis Testing – Testing relationships between trip features and fare amount
- Feature Engineering – Creating new features (e.g., trip duration, distance, location IDs)
- Model Building – Implemented and compared:
- Random Forest
- XGBoost
- GridSearchCV for hyperparameter tuning
- Evaluation – RMSE, MAE, and R² on test data
- Deployment – FastAPI backend + HTML/CSS frontend
- Testing – Automated tests with pytest
- CI/CD – GitHub Actions pipeline for build, test, and deploy
- Containerization – Dockerfile for reproducible builds and deploymen

## 🖥️ Frontend
- Built with HTML + CSS for a simple and clean UI
- Pages:
- Welcome Page – Introduction and navigation
- Prediction Page – Form to input trip details (duration, distance, pickup/dropoff IDs) and get fare prediction

## ⚙️ Backend
- FastAPI app (main.py) serves endpoints:
- / → Welcome page
- /predict → Form + prediction result
- Uvicorn used as ASGI server
- scripts/ folder contains:
- feature_engineering.py – Data preprocessing functions
- predict.py – Model loading and prediction logic

## 🚀 End-to-End Flow
- User opens frontend in browser
- Inputs trip details into form
- Request sent to FastAPI backend
- Backend calls ML model for prediction
- Predicted fare returned and displayed on frontend
- CI/CD ensures continuous testing and deployment
- Docker ensures reproducibility across environments


## 📁 Project Structure
```
├── data/
│   ├── raw/                  # Original dataset(s) from TLC
│   ├── processed/            # Cleaned & transformed datasets
│   └── external/             # External datasets if any (e.g. weather, holidays)
│
├── frontend/
│   ├── index.html
│   ├── predict.html
|
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_model_building.ipynb
│   └── 05_model_evaluation.ipynb
│
├── scripts/
│   ├── preprocessing.py          # Clean and transform data
│   ├── feature_engineering.py    # Feature engineering functions
│   ├── train_model.py            # Train and return model
|   ├── predict.py               # Predict fares
│   ├── evaluate.py         # Evaluate and save results
│   └── plot_utils.py             # Visualization and saving plots
│
├── models/                   # Saved ML models (e.g., .pkl files)
│
├── outputs/
│   ├── test_code.py          # Test Api
|
├── test/
│   ├── figures/              # Graphs and plots
│   └── reports/              # Exported results or summaries
│
├── .dockerignore             # Files/folders to ignore in docker
├── Dockerfile.txt            # Docker file to build docker image and containers
├── main.py                   # run and train the model
├── requirements.txt          # Project dependencies
├── README.md                 # Project overview and instructions
└── .gitignore                # Files/folders to ignore in Git
```

## 🔧 Installation

To run this project locally:

```bash
git clone https://github.com/mohsinraza2999/New-York-Taxi-Fare-Analysis.git
cd New-York-Taxi-Fare-Analysis
pip install -r requirements.txt
```
## 🚀 Usage

Run individual Jupyter notebooks in the notebooks/ directory to explore the analysis and modeling steps.

To Run a model from script:
``` bash
python main.py
```
Or build and run using docker
``` bash
docker build -t taxi-fare-app .
docker run -p 8000:8000 taxi-fare-app

```

## 📊 Sample Results

MAE: 11.16

R² Score: 0.86

(Update with real values after training models)

## ✅ Key Features
- End‑to‑end ML pipeline (EDA → Model → Deployment)
- FastAPI backend with REST endpoints
- Frontend UI for user interaction
- Dockerized for portability
- CI/CD pipeline for automation
- Unit tests with pytest

## 🤝 Acknowledgements

NYC Taxi & Limousine Commission (TLC) for the dataset

XGBoost, RandomForest, Scikit-learn, and Matplotlib, Seaborn for tools and libraries

## 📄 License

This project is for educational purposes only.
