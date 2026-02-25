# Data Splitting: Split the data into training and testing sets. Our target variable is silica_concentrate, located in the last column of the dataset. 
# This script will produce 4 datasets (X_test, X_train, y_test, y_train) that you can store in data/processed.  

# imports
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# make output folder if necessary
os.makedirs('data/processed_data', exist_ok=True)

# read data
data = pd.read_csv("data/raw_data/raw.csv")
X = data.drop("silica_concentrate", axis = 1)
y = data.silica_concentrate

# split
X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=42)

# save
X_train.to_csv("data/processed_data/X_train.csv")
X_test.to_csv("data/processed_data/X_test.csv")
y_train.to_csv("data/processed_data/y_train.csv")
y_test.to_csv("data/processed_data/y_test.csv")
