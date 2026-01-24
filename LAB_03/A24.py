import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
])

y = np.array([0, 0, 0, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = np.mean(knn_pred == y_test)

def matrix_inversion_predict(X_train, y_train, X_test):
    Xb_train = np.c_[np.ones(X_train.shape[0]), X_train]
    Xb_test = np.c_[np.ones(X_test.shape[0]), X_test]
    W = np.linalg.inv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train
    y_pred = Xb_test @ W
    return np.round(y_pred).astype(int)

mat_pred = matrix_inversion_predict(X_train, y_train, X_test)
mat_acc = np.mean(mat_pred == y_test)

print(knn_acc)
print(mat_acc)
