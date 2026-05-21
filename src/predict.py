# predict sample text
import joblib

# sample emails
emails = [
    "Congratulations! You won a free iPhone. Click now to claim reward.",
    
    "Hey Rohit, the meeting is scheduled tomorrow at 10 AM.",
    
    "URGENT: Your bank account has been suspended. Verify immediately.",
    
    "Can you send me the assignment PDF when free?",
    
    "Limited time offer! Earn money fast from home with zero investment."
]

# load trained model
model = joblib.load("models/spam_detector_svm.pkl")

# load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# convert emails into vectors
emails_vectorized = vectorizer.transform(emails)

# prediction
predictions = model.predict(emails_vectorized)

# output
for email, prediction in zip(emails, predictions):

    print("\n----------------------------------")
    print("Email:")
    print(email)

    print("\nPrediction:", prediction)

