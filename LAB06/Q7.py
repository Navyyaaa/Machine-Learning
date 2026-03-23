import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

file = input("Enter CSV file path: ").strip()
df = pd.read_csv(file)

target = input("Enter target column: ")
emb_col = input("Enter embedding column name: ").strip()

X = df[emb_col].apply(lambda x: np.array(eval(x)))
X = np.vstack(X)

y = df[target].values

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X2, y)

x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter(X2[:, 0], X2[:, 1], c=y)
plt.show()
