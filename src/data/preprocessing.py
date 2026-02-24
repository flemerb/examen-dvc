# Data Splitting: Split the data into training and testing sets. Our target variable is silica_concentrate, located in the last column of the dataset. 
# This script will produce 4 datasets (X_test, X_train, y_test, y_train) that you can store in data/processed.  

# read data
import pandas as pd

path_to_data = "./data/"
data = pd.read_csv(path_to_data + "raw_data/raw.csv")
X = data.drop("silica_concentrate", axis = 1)
y = data.silica_concentrate

# split
from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=42)

# save
X_train.to_csv(path_to_data + "processed_data/X_train.csv")
X_test.to_csv(path_to_data + "processed_data/X_test.csv")
y_train.to_csv(path_to_data + "processed_data/y_train.csv")
y_test.to_csv(path_to_data + "processed_data/y_test.csv")
