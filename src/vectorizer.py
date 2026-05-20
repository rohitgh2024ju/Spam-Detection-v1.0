# converting text-> vector
from preprocess import load_and_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

X, y = load_and_preprocess()

vectorizer = TfidfVectorizer(max_features=200000)

X_vectorized = vectorizer.fit_transform(X)
print(X_vectorized)
print(X_vectorized.shape)

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("vectorizer is saved successfully!")
