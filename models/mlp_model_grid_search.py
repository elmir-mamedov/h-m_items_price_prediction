import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor

# Load dataset
path = 'hm_preprocessed.csv'
data = pd.read_csv(path)

# Separate features and target
target_col = 'price'
X = data.drop(columns=[target_col])
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define Random Forest and parameter grid
mlp = MLPRegressor(random_state=42, early_stopping=True)
param_grid = {
    'hidden_layer_sizes': [(50, 20, 20), (50, 50, 10), (60, 30, 30)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001],
    'max_iter': [500, 1000  ]
}
# Setup GridSearchCV (using only X_train/y_train)
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

# Run grid search
print('Fitting grid search...')
grid_search.fit(X_train, y_train)

# Print best parameters and cross-validated score
print('\n=== Best Model ===')
print('Best parameters:', grid_search.best_params_)
print('Best R² (cross-validated):', grid_search.best_score_)

import json

# Save best params
with open('mlp_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f)