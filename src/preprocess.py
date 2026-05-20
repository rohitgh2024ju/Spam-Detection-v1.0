# text clearing
import pandas as pd

def load_and_preprocess():
    df = pd.read_csv("./data/cleaned_spam_data.csv")

    df = df.dropna(subset=["Message"])

    df["Subject"] = df["Subject"].fillna("")
    df["text"] = df["Subject"] + ": " + df["Message"]

    x = df["text"]
    y = df["Spam/Ham"]

    return x, y