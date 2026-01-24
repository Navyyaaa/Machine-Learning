import numpy as np
from collections import Counter
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
    X, y, test_size=0.3, random_state=42
)

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def custom_knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_neighbors = distances[:k]
        labels = [label for _, label in k_neighbors]
        predictions.append(Counter(labels).most_common(1)[0][0])
    return np.array(predictions)


y_pred_custom = custom_knn_predict(X_train, y_train, X_test, k=3)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_package = knn.predict(X_test)

print("Custom kNN Accuracy:", np.mean(y_pred_custom == y_test))
print("Package kNN Accuracy:", knn.score(X_test, y_test))
