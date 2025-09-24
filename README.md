# New-York-Taxi-Fare-Analysis
Analyzed NYC taxi fare data for TLC to build a fare estimation app. Conducted EDA, hypothesis testing, and regression analysis to identify key variables. Developed ML models (Random Forest, XGBoost) with GridSearchCV for hyperparameter tuning to predict fares accurately.
# NYC Taxi Fare Prediction 🚖

This project involves analyzing and modeling taxi fare data from New York City, provided by the NYC Taxi and Limousine Commission (TLC). The goal is to build a machine learning model that can accurately estimate taxi fares in advance for riders.

## 📌 Project Objective

In a fictional scenario, the NYC TLC has tasked the Automatidata team with building an app to help estimate taxi fares before a ride. This analysis aims to:

- Understand and explore the dataset
- Identify key variables affecting fare
- Conduct hypothesis testing and regression analysis
- Build predictive models using machine learning

## 🧪 Methodology

The project workflow includes:

1. **Data Understanding** – Reviewing raw data from TLC
2. **Exploratory Data Analysis (EDA)** – Identifying trends and anomalies
3. **Hypothesis Testing** – Testing relationships between variables
4. **Feature Engineering** – Creating and transforming useful features
5. **Model Building** – Using:
   - Random Forest
   - XGBoost
   - GridSearchCV for hyperparameter tuning
6. **Evaluation** – Assessing model performance on test data

## 📁 Project Structure

├── data/
│   ├── raw/                  # Original dataset(s) from TLC
│   ├── processed/            # Cleaned & transformed datasets
│   └── external/             # External datasets if any (e.g. weather, holidays)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_model_building.ipynb
│   └── 05_model_evaluation.ipynb
│
├── scripts/
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── train_model.py        # ML training pipeline
│   └── evaluate_model.py     # Evaluation & metrics
│
├── models/                   # Saved ML models (e.g., .pkl files)
│
├── outputs/
│   ├── figures/              # Graphs and plots
│   └── reports/              # Exported results or summaries
│
├── requirements.txt          # Project dependencies
├── README.md                 # Project overview and instructions
└── .gitignore                # Files/folders to ignore in Git


## 🔧 Installation

To run this project locally:

```bash
git clone https://github.com/yourusername/nyc-taxi-fare-prediction.git
cd nyc-taxi-fare-prediction
pip install -r requirements.txt
```
## 🚀 Usage

Run individual Jupyter notebooks in the notebooks/ directory to explore the analysis and modeling steps.

To train a model from script:
``` bash
python scripts/train_model.py
```
## 📊 Sample Results

MAE: X.XX

RMSE: X.XX

R² Score: 0.XX

(Update with real values after training models)

## 🤝 Acknowledgements

NYC Taxi & Limousine Commission (TLC) for the dataset

XGBoost, RandomForest, Scikit-learn, and Matplotlib, Seaborn for tools and libraries

## 📄 License

This project is for educational purposes only.
