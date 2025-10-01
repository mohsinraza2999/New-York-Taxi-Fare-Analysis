import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
from scripts.preprocessing import load_and_clean_data
from scripts.feature_engineering import add_features
import pandas as pd

#this lines of code is used to test preprocessing and feature engineering scripts to test these first 
#put data to data/raw folder or put the data path
df = pd.read_csv("data/raw/dataset_name.csv or data_path")
sample1=df.sample(n=100, replace=True, random_state=42)
sample2=df.sample(n=100, replace=True, random_state=40)
sample3=df.sample(n=100, replace=True, random_state=24)
def test_preprocessing_returns_tuple():
    
    assert load_and_clean_data(sample1) is not None
    assert load_and_clean_data(sample2) is not None
    assert load_and_clean_data(sample3) is not None

def test_feature_engineering():
    assert add_features(sample1) is not None
    assert add_features(sample2) is not None
    assert add_features(sample3) is not None"""

from main import app
from fastapi.testclient import TestClient
client=TestClient(app)

def test_api():
    response=client.get('/')
    assert response.status_code==200