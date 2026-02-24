# Data Normalization: As you may notice, the data varies widely in scale, so normalization is necessary. 
# You can use existing functions to construct this script. As output, this script will create two new datasets 
# (X_train_scaled, X_test_scaled) which you will also save in data/processed.

# data normalization
import pandas as pd
data_path = "./data/processed_data/"
X_train = pd.read_csv(data_path + "X_train.csv")
X_test = pd.read_csv(data_path + "X_test.csv")

# standardization
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.drop(["date"], axis = 1)))
X_test_scaled = pd.DataFrame(scaler.transform(X_test.drop(["date"], axis = 1)))

# save
X_train_scaled.to_csv(data_path + "X_train_scaled.csv")
X_test_scaled.to_csv(data_path + "X_test_scaled.csv")

