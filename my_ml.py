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
#print(comparison.head())

# Calculating the mean absolute error in validation data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
#print(val_mae)

# Underfitting and Overfitting
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
# dictionary to store max_leaf_nodes and their corresponding MAE
select_best_value = {}
for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    select_best_value[max_leaf_nodes] = my_mae
    print("Max leaf nodes {} \t Mean Absolute Error: {}".format(max_leaf_nodes, my_mae))

# The min() function is used with mae_values.get to find
# the key (i.e., max_leaf_nodes) that has the smallest MAE.
best_tree_size = min(select_best_value, key = select_best_value.get)

#print("Best Tree Leaf Node {} \t Lowest MAE {}".format(best_tree_size, select_best_value[best_tree_size]))

# Now that we know the best value for max_leaf_nodes
# we do not need to hold out a validation set
# and we can retrain the model on ALL the data

final_X = pd.concat([train_x, val_x])
final_Y = pd.concat([train_y, val_y])

final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 0)
final_model.fit(final_X, final_Y)