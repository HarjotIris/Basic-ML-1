# import pandas
import pandas as pd

# save filepath to variable for easier access
iowa_file_path = r"C:\Users\HP\train.csv"
# read the data and store the data in DataFrame called home_data
home_data = pd.read_csv(iowa_file_path)

''' 
Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
'''
"""
print("Setup complete")
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

# Import the DecisionTreeRegressor class from sklearn
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(x, y)
predictions = iowa_model.predict(x)
#print(predictions[:5])

comparison = pd.DataFrame({"Actual": y, "Predicted": predictions})
print(comparison.head())
