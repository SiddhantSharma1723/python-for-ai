'''
1 import all required modules
2 import dataset
3 choose features
4 choose target_column
5 import splitter.py
6 split train val and test
7 import baseline.py
8 establish a baseline
9 train linear regression model
10 find mae and rmse
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import(mean_absolute_error, root_mean_squared_error)

data = pd.read_csv("../../Data/house_price_practice.csv")
features = ["size_sqft","bedrooms","age_years","distance_km"]
target_column = "price"

train_data, val_data, test_data = split_dataset(data)

X_train = train_data[features]
y_train = train_data[target_column]
X_val = val_data[features]
y_val = val_data[target_column]

baseline_mae, baseline_rmse = mean_baseline(
    "House Price", train_data, val_data, target_column
)

model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = root_mean_squared_error(y_val, predictions)

print(f"\n\nMAE for Model is {mae:.2f}\nRMSE for Model is {rmse:.2f}\n\n")


if mae<baseline_mae and rmse<baseline_rmse:
    print("Model has cleared the Baseline criteria for both MAE and RMSE.")
else:
    print("The model did'nt cross the Baseline.")

print(model.coef_)
print(model.intercept_)

train_predictions = model.predict(X_train)

train_mae = mean_absolute_error(train_predictions,y_train)
train_rmse = root_mean_squared_error(train_predictions,y_train)

print(f"Training MAE is {train_mae:.2f}")
print(f"Training RMSE is {train_rmse:.2f}")

print(f"Validation MAE is {mae:.2f}")
print(f"Validation RMSE is {rmse:.2f}")

mae_gap = mae - train_mae
rmse_gap = rmse - train_rmse

print("\n\nMAE gap is : ",mae_gap)
print("RMSE gap is : ",rmse_gap)

train_residual = y_train - train_predictions
val_residual = y_val - predictions

plt.scatter(predictions,val_residual)
plt.axhline(0)
plt.xlabel("Predicted Price")
plt.ylabel("Residual")