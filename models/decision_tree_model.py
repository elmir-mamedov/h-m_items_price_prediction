import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv("hm_preprocessed.csv")

# Target and features
X = df.drop("price", axis=1)
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
tree = DecisionTreeRegressor(random_state=42)
param_grid = {
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(
    tree,
    param_grid,
    scoring="r2",
    cv=5,
    verbose=2,
    n_jobs=-1
)

print("Training model...")
grid_search.fit(X_train, y_train)

# Print best score and best parameters
print('Best parameters:', grid_search.best_params_)
print('Best R²:', grid_search.best_score_)

# Save best parameters to json
with open('decision_tree_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f)