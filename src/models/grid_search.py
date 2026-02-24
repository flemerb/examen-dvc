# GridSearch for Best Parameters: 
# Decide on the regression model to implement and the parameters to test. At the end of this script, 
# we will have the best parameters saved as a .pkl file in the models directory.

# gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# data
import pandas as pd
data_path = "./data/processed_data/"
model_path = "./models/"
X_train_scaled = pd.read_csv(data_path + "X_train_scaled.csv")
y_train = pd.read_csv(data_path + "y_train.csv")

param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
ridge_best_params = grid_search.best_params_ 


# save model
import joblib
joblib.dump(ridge_best_params, model_path+'ridge_best_params.pkl')

# Load
ridge_best_params = joblib.load(model_path+'ridge_best_params.pkl')