import pandas as pd
import numpy as np

# ============================
# FUNCTION: Load Dataset
# ============================
def load_dataset(file_path):
    """
    Load dataset, detect label column, remove text columns.
    """
    data_frame = pd.read_csv(file_path)

    # Detect label column
    possible_labels = ['Label', 'label', 'target', 'class', 'output']
    label_column = None

    for col in possible_labels:
        if col in data_frame.columns:
            label_column = col
            break

    if label_column is None:
        label_column = data_frame.columns[-1]

    # Keep only numeric columns (remove text columns like Sentence)
    numeric_df = data_frame.select_dtypes(include=[np.number])

    # Ensure label column exists
    if label_column not in numeric_df.columns:
        numeric_df[label_column] = data_frame[label_column]

    # Features and labels
    feature_matrix = numeric_df.drop(label_column, axis=1).values
    label_vector = numeric_df[label_column].values

    return feature_matrix, label_vector, label_column


# ============================
# FUNCTION: Split Dataset
# ============================
def split_dataset(X, y):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ============================
# FUNCTION: Create Stacking Model (A1)
# ============================
def create_stacking_model():
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB

    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC(probability=True)),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('nb', GaussianNB())
    ]

    meta_model = LogisticRegression()

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    return stacking_model


# ============================
# FUNCTION: Create Pipeline (A2)
# ============================
def create_pipeline(stacking_model):
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('classifier', stacking_model)
    ])

    return pipeline


# ============================
# FUNCTION: Train Model
# ============================
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


# ============================
# FUNCTION: Evaluate Model
# ============================
def evaluate_model(pipeline, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix


# ============================
# FUNCTION: Plot Confusion Matrix
# ============================
def plot_confusion_matrix(matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(matrix, annot=True, fmt='d')
    plt.title("Confusion Matrix (Stacking Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ============================
# FUNCTION: LIME Explanation (A3)
# ============================
def explain_with_lime(pipeline, X_train, X_test, index=0):
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train,
        mode='classification',
        feature_names=[f'f{i}' for i in range(X_train.shape[1])],
        class_names=["Incoherent", "Coherent"],
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(
        X_test[index],
        pipeline.predict_proba,
        num_features=10
    )

    return explanation


# ============================
# MAIN PROGRAM
# ============================
if __name__ == "__main__":

    file_path = "D:/Sem4/ml/lab/assignments/done/Lab09/Coherence_bert_cls_embeddings.csv"

    # Load dataset
    X, y, label_column = load_dataset(file_path)
    print("Label column detected:", label_column)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Create stacking model (A1)
    stacking_model = create_stacking_model()

    # Create pipeline (A2)
    pipeline = create_pipeline(stacking_model)

    # Train model
    trained_model = train_model(pipeline, X_train, y_train)
    print("Model training completed.")

    # Evaluate model
    accuracy, report, matrix = evaluate_model(trained_model, X_test, y_test)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)

    # Plot confusion matrix
    plot_confusion_matrix(matrix)

    # LIME explanation (A3)
    lime_exp = explain_with_lime(trained_model, X_train, X_test, index=0)

    print("\nLIME Explanation:")
    for feature, weight in lime_exp.as_list():
        print(feature, ":", weight)

    import matplotlib.pyplot as plt
    fig = lime_exp.as_pyplot_figure()
    plt.show()
