import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv(r"D:\Sem4\ml\lab\assignments\done\LAB05\projectds.csv")
data["embedding"] = data["embedding"].apply(ast.literal_eval)
X = pd.DataFrame(data["embedding"].to_list())
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2_test = r2_score(y_test, y_test_pred)
print("TRAIN METRICS (All Attributes)")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R2:", r2_train)

print("\nTEST METRICS (All Attributes)")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R2:", r2_test)
