import pandas as pd
import numpy as np

def load_and_clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() // 60
    df['fare_amount'] = np.where(df['fare_amount'] < 0, 0, df['fare_amount'])
    df['duration'] = np.where(df['duration'] < 0, 0, df['duration'])

    # Cap fare_amount outliers
    q1 = df['fare_amount'].quantile(0.25)
    q3 = df['fare_amount'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 6 * iqr
    df['fare_amount'] = np.where(df['fare_amount'] > upper_limit, upper_limit, df['fare_amount'])

    # Cap duration outliers
    q1 = df['duration'].quantile(0.25)
    q3 = df['duration'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 6 * iqr
    df['duration'] = np.where(df['duration'] > upper_limit, 0, df['duration'])

    return df