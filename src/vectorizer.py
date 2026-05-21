# converting text-> vector
from preprocess import load_and_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import joblib


def vectorize_dataset():
    X, y = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=20000)

    X_train_vectorized = vectorizer.fit_transform(X_train)
    print(X_train_vectorized.shape)

    X_test_vectorized = vectorizer.transform(X_test)
    print(X_test_vectorized.shape)

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("vectorizer is saved successfully!")
    return (X_train_vectorized, X_test_vectorized, y_train, y_test)


vectorize_dataset()
