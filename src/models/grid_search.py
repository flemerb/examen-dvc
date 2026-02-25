# GridSearch for Best Parameters: 
# Decide on the regression model to implement and the parameters to test. At the end of this script, 
# we will have the best parameters saved as a .pkl file in the models directory.

# gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# data
import pandas as pd
X_train_scaled = pd.read_csv("data/normalized_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
ridge_best_params = grid_search.best_params_ 

# save model
import joblib
joblib.dump(ridge_best_params, 'models/best_params.pkl')
