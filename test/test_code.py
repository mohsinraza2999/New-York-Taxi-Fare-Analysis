from scripts.preprocessing import load_and_clean_data
from scripts.feature_engineering import add_features
import pandas as pd
df = pd.read_csv("data/raw/2017_Yellow_Taxi_Trip_Data.csv")
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
    assert add_features(sample3) is not None