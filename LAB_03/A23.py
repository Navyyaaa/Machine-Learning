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

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

def confusion_matrix_custom(y_true, y_pred):
    TP = FP = FN = TN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN

def accuracy(TP, FP, FN, TN):
    return (TP + TN) / (TP + FP + FN + TN)

def precision(TP, FP):
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def fbeta(prec, rec, beta=1):
    if prec + rec == 0:
        return 0
    return (1 + beta**2) * prec * rec / ((beta**2 * prec) + rec)

TP, FP, FN, TN = confusion_matrix_custom(y_test, y_pred_test)

prec = precision(TP, FP)
rec = recall(TP, FN)
f1 = fbeta(prec, rec, beta=1)
acc = accuracy(TP, FP, FN, TN)

print(TP, FP, FN, TN)
print(acc)
print(prec)
print(rec)
print(f1)
