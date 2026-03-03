import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r"D:\Sem4\ml\lab\assignments\done\LAB05\projectds.csv")
data["embedding"] = data["embedding"].apply(ast.literal_eval)
X_full = pd.DataFrame(data["embedding"].to_list())
X = X_full.iloc[:, [0]]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
print("Model trained successfully")
