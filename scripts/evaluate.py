import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_model(model, X_test, y_test, output_dir='outputs/reports/'):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test.values - y_pred

    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Residual': residuals
    })
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/regression_results.csv", index=False)

    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump({'mse': mse, 'r2': r2}, f, indent=4)

    return results_df, mse, r2