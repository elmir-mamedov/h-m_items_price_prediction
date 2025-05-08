import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
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

# Vectorization of "details" column
text_columns = ["details"]
numeric_columns = [col for col in X.columns if col not in text_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=1000), 'details'),
        ('num', StandardScaler(), numeric_columns)
    ])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'SVR': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'MLP': MLPRegressor(max_iter=1000, random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'kNN': KNeighborsRegressor(),
    'XGBoost': XGBRegressor()
}

param_grids = {
    'SVR': {
        'reg__C': [10],
        'reg__kernel': ["rbf"],
        'reg__epsilon': [0.1]
    },
    'Decision Tree': {
        'reg__max_depth': [15],
        'reg__min_samples_split': [5]
    },
    'MLP': {
        'reg__hidden_layer_sizes': [64],
        'reg__activation': ["relu"],
        'reg__learning_rate_init': [0.001],
        'reg__solver': ["adam"]

    },
    'Random Forest': {
        'reg__n_estimators': [20]
    },
    'kNN': {
        'reg__n_neighbors': [3],
        'reg__weights': ["distance"],
        'reg__algorithm': ["auto"]       
    },
    'XGBoost': {
        'reg__learning_rate': [0.3],
        'reg__max_depth': [7],
        'reg__n_estimators': [200],
        'reg__subsample': [1.0]
    }
}

# Initialize results storage
mse_vals, rmse_vals, r2_vals, best_params = [], [], [], []

# Perform GridSearchCV for each model
for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('reg', model)
    ])
    
    # GridSearchCV setup
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store results
    mse_vals.append(mse)
    rmse_vals.append(rmse)
    r2_vals.append(r2)
    best_params.append(grid_search.best_params_)

# Convert results to DataFrame for easy visualization
results_df = pd.DataFrame({
    'Model': list(models.keys()),
    'MSE': mse_vals,
    'RMSE': rmse_vals,
    'R^2': r2_vals,
    'Best Parameters': best_params
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