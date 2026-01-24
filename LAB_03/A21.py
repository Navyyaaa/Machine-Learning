import numpy as np
import matplotlib.pyplot as plt
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
    X, y, test_size=0.3, random_state=1
)

max_k = len(X_train)
k_values = range(1, max_k + 1)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.show()

print(accuracies[0])
print(accuracies[2] if len(accuracies) > 2 else accuracies[-1])
