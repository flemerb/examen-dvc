# Data Normalization: As you may notice, the data varies widely in scale, so normalization is necessary. 
# You can use existing functions to construct this script. As output, this script will create two new datasets 
# (X_train_scaled, X_test_scaled) which you will also save in data/processed.

# imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import os  

# make output folder if necessary
os.makedirs('data/normalized_data', exist_ok=True)

# get data
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# MinMaxScaling
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.drop(["date"], axis = 1)))
X_test_scaled = pd.DataFrame(scaler.transform(X_test.drop(["date"], axis = 1)))

# save
X_train_scaled.to_csv("data/normalized_data/X_train_scaled.csv")
X_test_scaled.to_csv("data/normalized_data/X_test_scaled.csv")

