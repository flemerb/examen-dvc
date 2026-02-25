# Train Ridge with best parameters
from sklearn.linear_model import Ridge
import joblib
import pandas as pd

# get best parameters
best_params = joblib.load("models/best_params.pkl")

# get data
X_train_scaled = pd.read_csv("data/normalized_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

best_model = Ridge(**best_params)
best_model.fit(X_train_scaled, y_train)

# Save to file
joblib.dump(best_model, 'models/best_model.pkl')
