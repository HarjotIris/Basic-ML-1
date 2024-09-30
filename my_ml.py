# Code you have previously used to load data
import pandas as pd

# Code you have previously used to load data
iowa_file_path = r"C:\Users\HP\train.csv"

home_data = pd.read_csv(iowa_file_path)


"""
print(home_data.describe())
print(home_data.columns)
"""
y = home_data['SalePrice']
#print(y)


feature_names = ['LotArea', 'YearBuilt', 
                 '1stFlrSF', '2ndFlrSF', 
                 'FullBath', 'BedroomAbvGr', 
                 'TotRmsAbvGrd']

x = home_data[feature_names]

#print(x['LotArea'], x['YearBuilt'])

#Import the DecisionTreeRegressor class from sklearn
from sklearn.tree import DecisionTreeRegressor
"""
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(x, y)
predictions = iowa_model.predict(x)
print(predictions[:5])

comparison = pd.DataFrame( {"Actual": y, "Predicted": predictions})
print(comparison.head())
"""

# Model Validation begins
from sklearn.model_selection import train_test_split
train_x, val_x, train_y,val_y = train_test_split(x, y, random_state = 1)
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(train_x, train_y)

val_predictions = iowa_model.predict(val_x)

comparison = pd.DataFrame({"Actual": val_y, "Predicted": val_predictions})
print(comparison.head())

# Calculating the mean absolute error in validation data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
