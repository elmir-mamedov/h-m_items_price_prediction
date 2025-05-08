import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

# Load dataset
df = pd.read_csv("hm_preprocessed.csv")

# Target and features
X = df.drop("price", axis=1)
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Regressor
svr = SVR()
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "epsilon": [0.1, 0.2]
}

# GridSearchCV
grid_search = GridSearchCV(
    svr,
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
with open('svr_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f)