import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
# Load model
best_model = joblib.load('models/best_model.pkl')

# load data
X_test_scaled = pd.read_csv("data/normalized_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

# Make predictions
y_pred_test = best_model.predict(X_test_scaled)

# Save predictions
np.savetxt('data/predictions.csv', y_pred_test, delimiter = ',')

# Compute metrics
scores = {
    'MSE': mean_squared_error(y_test, y_pred_test),
    'R2': r2_score(y_test, y_pred_test)
}

# Save metrics
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print(scores)