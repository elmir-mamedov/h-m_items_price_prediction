import json
from sklearn.neural_network import MLPRegressor

# 1. Load the saved parameters
with open('mlp_params.json', 'r') as f:
    params = json.load(f)

# 2. Create the model
MLP_model = MLPRegressor(**params)
