import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_model(df, save_model_path='model/reg_model.pkl'):
    df = df[['fare_amount', 'trip_distance', 'mean_distance', 'duration']].dropna()

    X = df[['trip_distance', 'mean_distance', 'duration']]
    y = df['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, save_model_path)
    joblib.dump(scaler, 'model/scaler.pkl')

    return model, scaler, X_test_scaled, y_test