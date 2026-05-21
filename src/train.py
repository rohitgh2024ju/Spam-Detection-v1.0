# actual training
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vectorizer import vectorize_dataset


def model_train(evaluation=False):
    # load configs
    X_train_vectorized, X_test_vectorized, y_train, y_test = vectorize_dataset()

    # load model
    model = LinearSVC(dual=False, random_state=42)

    # train
    model.fit(X_train_vectorized, y_train)

    # prediction
    predictions = model.predict(X_test_vectorized)
    print(predictions[:10])

    if evaluation:
        # evaluation
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: ", accuracy)

        report = classification_report(y_test, predictions)
        print("\n--Classification Report--\n", report)

        confused_matrix = confusion_matrix(y_test, predictions)
        print("\n--Confusion Matrix--\n", confused_matrix)


if __name__ == "__main__":
    model_train(evaluation=True)
