# prediction function
import joblib

# load vectorizer and model
try:
    VECTORIZER = joblib.load("./models/tfidf_vectorizer.pkl")
    MODEL = joblib.load("./models/spam_detector_svm.pkl")
except FileNotFoundError:
    print("Warning: Model files not found. Ensure you run train.py first.")


def predict_mail(mail):
    # Defensive check for empty inputs
    if not mail or not str(mail).strip():
        return None

    mail_vectorized = VECTORIZER.transform([str(mail)])

    # Predict
    prediction = MODEL.predict(mail_vectorized)[0]

    # confidence parameter
    distance = MODEL.decision_function(mail_vectorized)[0]

    # set confidence
    if abs(distance) > 2.0:
        confidence = "very high"
    elif abs(distance) > 1.0:
        confidence = "high"
    else:
        confidence = "borderline"

    return (mail, prediction, distance, confidence)


if __name__ == "__main__":
    test_email = (
        "Congratulations! You've won a free $1000 gift card. Click here to claim now."
    )
    result = predict_mail(test_email)
    if result:
        print(f"Email: {result[0]}")
        print(f"Prediction: {result[1]}")
        print(f"Confidance Score : {result[2]:.2f} ")
