import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json

from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Silence warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
df = pd.read_csv("hm_preprocessed.csv")

# Target and features
X = df.drop("price", axis=1)
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
with open('svr_params.json', 'r') as f:
    svr_params = json.load(f)
with open('decision_tree_params.json', 'r') as f:
    dt_params = json.load(f)
with open('random_forest_params.json', 'r') as f:
    rf_params = json.load(f)
with open('mlp_params.json', 'r') as f:
    mlp_params = json.load(f)
with open('best_params_xgboost.json', 'r') as f:
    xgb_params = json.load(f)
xgb_params = xgb_params['XGBoost']
with open('best_params_knn.json', 'r') as f:
    knn_params = json.load(f)
knn_params = knn_params['kNN']


# Create models with loaded parameters
models = {
    'SVR': SVR(**svr_params),
    'Decision Tree': DecisionTreeRegressor(random_state=42, **dt_params),
    'Random Forest': RandomForestRegressor(random_state=42, **rf_params),
    'MLP': MLPRegressor(random_state=42, **mlp_params),
    'XGBoost': XGBRegressor(random_state=42, **xgb_params),
    'kNN': KNeighborsRegressor(**knn_params)
}

# Initialize results storage
mse_vals, rmse_vals, r2_vals, best_params = [], [], [], []

# Perform GridSearchCV for each model
for model_name, model in models.items():
    print(f"Fitting {model_name}...")

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store results
    mse_vals.append(mse)
    rmse_vals.append(rmse)
    r2_vals.append(r2)

# Convert results to DataFrame for easy visualization
results_df = pd.DataFrame({
    'Model': list(models.keys()),
    'MSE': mse_vals,
    'RMSE': rmse_vals,
    'R^2': r2_vals,
})

print(results_df.sort_values('RMSE'))
results_df.to_csv('model_comparison_results.csv', index=False)

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(18, 5))

# MSE graph
plt.subplot(1, 3, 1)
sns.barplot(x='Model', y='MSE', data=results_df, hue='Model', palette='Oranges_d', legend=False)
plt.title('MSE by Model')
plt.ylabel('MSE')
plt.xticks(rotation=45)

# RMSE graph
plt.subplot(1, 3, 2)
sns.barplot(x='Model', y='RMSE', data=results_df, hue='Model', palette='Blues_d', legend=False)
plt.title('RMSE by Model')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

# R^2 graph
plt.subplot(1, 3, 3)
sns.barplot(x='Model', y='R^2', data=results_df, hue='Model', palette='Greens_d', legend=False)
plt.title('R² Score by Model')
plt.ylabel('R²')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('model_comparison_results.png')
plt.show()