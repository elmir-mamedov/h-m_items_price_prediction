import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

'''
Find RandomForest best parameters from grid_search
Save the parameters in json file 
'''


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
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 100, 1000],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt']
}

# Setup GridSearchCV (using only X_train/y_train)
grid_search = GridSearchCV(
    estimator=rf,
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
with open('random_forest_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f)
