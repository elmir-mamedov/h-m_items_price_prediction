import json
from sklearn.ensemble import RandomForestRegressor

# 1. Load the saved parameters
with open('random_forest_params.json', 'r') as f:
    params = json.load(f)

# 2. Create the model
RandomForest_model = RandomForestRegressor(**params)
