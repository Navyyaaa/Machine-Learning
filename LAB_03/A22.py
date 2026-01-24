import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(cm_train)
print(precision_score(y_train, y_train_pred, zero_division=0))
print(recall_score(y_train, y_train_pred, zero_division=0))
print(f1_score(y_train, y_train_pred, zero_division=0))

print(cm_test)
print(precision_score(y_test, y_test_pred, zero_division=0))
print(recall_score(y_test, y_test_pred, zero_division=0))
print(f1_score(y_test, y_test_pred, zero_division=0))

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(train_acc)
print(test_acc)

if abs(train_acc - test_acc) < 0.1:
    print("Regular Fit")
elif train_acc > test_acc:
    print("Overfit")
else:
    print("Underfit")
