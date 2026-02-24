import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

data_path = "./data/processed_data/"
model_path = "./models/"

# Load model
ridge_best_model = joblib.load(model_path+'ridge_best_model.pkl')

# load data
X_test_scaled = pd.read_csv(data_path + "X_test_scaled.csv")
y_test = pd.read_csv(data_path + "y_test.csv")

# Make predictions
y_pred_test = ridge_best_model.predict(X_test_scaled)

# Save predictions
np.savetxt(data_path+'predictions.csv', y_pred_test, delimiter = ',')

# Compute metrics
scores = {
    'MSE': mean_squared_error(y_test, y_pred_test),
    'R2': r2_score(y_test, y_pred_test)
}

# Save metrics
with open('./metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print(scores)