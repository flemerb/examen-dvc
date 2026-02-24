# Train Ridge with best parameters
from sklearn.linear_model import Ridge
import joblib
import pandas as pd

# get best parameters
data_path = "./data/processed_data/"
model_path = "./models/"
ridge_best_params = joblib.load(model_path+'ridge_best_params.pkl')

# get data
X_train_scaled = pd.read_csv(data_path + "X_train_scaled.csv")
y_train = pd.read_csv(data_path + "y_train.csv")

ridge_best_model = Ridge(**ridge_best_params)
ridge_best_model.fit(X_train_scaled, y_train)

# Save to file
joblib.dump(ridge_best_model, model_path+'ridge_best_model.pkl')
