import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

file = input("Enter CSV file path: ").strip()
df = pd.read_csv(file)

print("Columns:", df.columns)

target = input("Enter target column: ")
emb_col = input("Enter embedding column name: ")

X = df[emb_col].apply(lambda x: np.array(eval(x)))
X = np.vstack(X)

y = df[target].values

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

plt.figure(figsize=(12,6))
plot_tree(model, filled=True, max_depth=3)
plt.show()
