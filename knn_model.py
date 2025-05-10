from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
import pandas as pd
import json

!pip install xgboost

uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))


target_column = "price"
X = df.drop(target_column, axis=1)
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsRegressor()
knn_params = {
    "n_neighbors": [3, 5, 7],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
}
knn_grid = GridSearchCV(knn, knn_params, scoring="neg_mean_squared_error", cv=KFold(n_splits=5, shuffle=True, random_state=42))
knn_grid.fit(X_train_scaled, y_train)
knn_best = knn_grid.best_estimator_


knn_pred = knn_best.predict(X_test_scaled)

knn_mse = mean_squared_error(y_test, knn_pred)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, knn_pred)

best_params = {
    "kNN": knn_grid.best_params_,
}


#  results
print("kNN MSE:", knn_mse)
print("kNN RMSE:", knn_rmse)
print("kNN R2:", knn_r2)

with open("best_params.json", "w") as json_file:
    json.dump(best_params, json_file, indent=4)

files.download('best_params.json')
