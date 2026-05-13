import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    text = " ".join(
        word for word in text.split()
        if word not in STOP_WORDS
    )

    return text


def main():

    df = pd.read_json("file.json", lines=True)

    df = df[["headline", "category"]].dropna()

    df.drop_duplicates(subset=["headline"], inplace=True)

    counts = df["category"].value_counts()

    valid_categories = counts[counts >= 2000].index

    df = df[df["category"].isin(valid_categories)]

    df["headline"] = df["headline"].apply(clean)

    X = df["headline"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000,
        C=5,
        solver="lbfgs",
        multi_class="multinomial"
    )

    lr.fit(X_train_vec, y_train)

    nb = MultinomialNB()

    nb.fit(X_train_vec, y_train)

    lr_pred = lr.predict(X_test_vec)
    nb_pred = nb.predict(X_test_vec)

    lr_acc = accuracy_score(y_test, lr_pred)
    nb_acc = accuracy_score(y_test, nb_pred)

    print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
    print(f"Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

    print(classification_report(y_test, lr_pred))

    print(confusion_matrix(y_test, lr_pred))

    with open("news_model.pkl", "wb") as f:
        pickle.dump(lr, f)

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)


if __name__ == "__main__":
    main()