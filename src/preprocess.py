# text clearing
import pandas as pd


def load_and_preprocess():
    # read dataset
    df = pd.read_csv("./data/combined_data.csv")

    # transform the labels
    df["label"] = df["label"].map({1: "spam", 0: "ham"})

    # drop all rigged rows
    df = df.dropna(subset=["text"])

    # enforce text as string
    df["text"] = df["text"].astype(str)

    # drop all duplicate rows
    df = df.drop_duplicates(subset=["text"])

    x = df["text"]
    y = df["label"]

    return x, y
