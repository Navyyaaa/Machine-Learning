import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski
import dataset

def dot_product(vector1, vector2):
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    return result

def euclidean_norm(vector):
    sum_sq = 0
    for value in vector:
        sum_sq += value * value
    return sum_sq ** 0.5

def calculate_mean(values):
    total = 0
    for value in values:
        total += value
    return total / len(values)

def calculate_variance(values, mean_value):
    total = 0
    for value in values:
        total += (value - mean_value) ** 2
    return total / len(values)

def minkowski_distance(vector1, vector2, p_value):
    total = 0
    for i in range(len(vector1)):
        total += abs(vector1[i] - vector2[i]) ** p_value
    return total ** (1 / p_value)

def custom_knn_predict(train_data, train_labels, test_vector, k_value):
    distance_list = []
    for i in range(len(train_data)):
        distance = euclidean_norm(train_data[i] - test_vector)
        distance_list.append((distance, train_labels[i]))

    distance_list.sort(key=lambda x: x[0])

    vote_count = {}
    for i in range(k_value):
        label = distance_list[i][1]
        vote_count[label] = vote_count.get(label, 0) + 1

    return max(vote_count, key=vote_count.get)

def calculate_confusion_metrics(actual_labels, predicted_labels):
    true_pos = true_neg = false_pos = false_neg = 0

    for i in range(len(actual_labels)):
        if actual_labels[i] == 1 and predicted_labels[i] == 1:
            true_pos += 1
        elif actual_labels[i] == 0 and predicted_labels[i] == 0:
            true_neg += 1
        elif actual_labels[i] == 0 and predicted_labels[i] == 1:
            false_pos += 1
        elif actual_labels[i] == 1 and predicted_labels[i] == 0:
            false_neg += 1

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) != 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return true_pos, true_neg, false_pos, false_neg, accuracy, precision, recall, f1_score

def main():
    data_frame = dataset.data
    feature_matrix = data_frame.iloc[:, :-1].values
    class_labels = data_frame.iloc[:, -1].values

    vector_A = feature_matrix[0]
    vector_B = feature_matrix[1]

    manual_dot = dot_product(vector_A, vector_B)
    numpy_dot = np.dot(vector_A, vector_B)

    manual_norm = euclidean_norm(vector_A)
    numpy_norm = np.linalg.norm(vector_A)

    class_zero_data = feature_matrix[class_labels == 0]
    class_one_data = feature_matrix[class_labels == 1]

    mean_class_zero = np.mean(class_zero_data, axis=0)
    mean_class_one = np.mean(class_one_data, axis=0)

    std_class_zero = np.std(class_zero_data, axis=0)
    std_class_one = np.std(class_one_data, axis=0)

    interclass_distance = np.linalg.norm(mean_class_zero - mean_class_one)

    selected_feature = feature_matrix[:, 0]
    histogram_values, bin_edges = np.histogram(selected_feature, bins=5)

    feature_mean = calculate_mean(selected_feature)
    feature_variance = calculate_variance(selected_feature, feature_mean)

    plt.hist(selected_feature, bins=5)
    plt.show()

    minkowski_values = []
    for p in range(1, 11):
        minkowski_values.append(minkowski_distance(vector_A, vector_B, p))

    plt.plot(range(1, 11), minkowski_values)
    plt.show()

    scipy_minkowski = minkowski(vector_A, vector_B, 3)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, class_labels, test_size=0.3, random_state=1
    )

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    knn_accuracy = knn_model.score(X_test, y_test)
    sklearn_predictions = knn_model.predict(X_test)

    custom_predictions = []
    for test_vector in X_test:
        custom_predictions.append(custom_knn_predict(X_train, y_train, test_vector, 3))

    k_list = []
    accuracy_list = []
    max_k = len(X_train)

    for k in range(1, max_k + 1):
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train, y_train)
        k_list.append(k)
        accuracy_list.append(temp_model.score(X_test, y_test))

    true_pos, true_neg, false_pos, false_neg, acc, prec, rec, f1 = calculate_confusion_metrics(
        y_test, sklearn_predictions
    )

    print("A1:", manual_dot, numpy_dot, manual_norm, numpy_norm)
    print("A2:", mean_class_zero, mean_class_one, std_class_zero, std_class_one, interclass_distance)
    print("A3:", feature_mean, feature_variance)
    print("A4 Minkowski distances:", minkowski_values)
    print("A5 SciPy Minkowski:", scipy_minkowski)
    print("A6 Train size:", len(X_train), "Test size:", len(X_test))
    print("A7 kNN trained with k=3")
    print("A8 Accuracy:", knn_accuracy)
    print("A9 Predictions (sklearn):", sklearn_predictions)
    print("A10 Predictions (custom):", custom_predictions)
    print("A11 k values:", k_list)
    print("A11 Accuracies:", accuracy_list)
    print("A12/A13:", true_pos, true_neg, false_pos, false_neg, acc, prec, rec, f1)
    print("A14 Comparison: kNN is distance-based, matrix inversion assumes linear separability")

main()
