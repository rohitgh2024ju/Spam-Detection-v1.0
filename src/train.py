# actual training
import joblib
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

    joblib.dump(model, "models/spam_detector_svm.pkl")
    print("Model saved successfully to 'models/spam_detector_svm.pkl'!")

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

"""
(26931, 20000)
(6733, 20000)
vectorizer is saved successfully!
(26931, 20000)
(6733, 20000)
vectorizer is saved successfully!
['spam' 'spam' 'ham' 'ham' 'spam' 'ham' 'ham' 'ham' 'spam' 'ham']
Accuracy:  0.999405911183722

--Classification Report--
               precision    recall  f1-score   support

         ham       1.00      1.00      1.00      3269
        spam       1.00      1.00      1.00      3464

    accuracy                           1.00      6733
   macro avg       1.00      1.00      1.00      6733
weighted avg       1.00      1.00      1.00      6733


--Confusion Matrix--
 [[3265    4]
 [   0 3464]]

"""
