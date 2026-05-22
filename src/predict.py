import joblib

def predict_mail(mail):
    if not mail:
        return None
    
    # load vectorizer
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

    # load model
    model = joblib.load('./models/spam_detector_svm.pkl')

    # vectorize the mail
    mail_vectorized = vectorizer.transform(mail.)