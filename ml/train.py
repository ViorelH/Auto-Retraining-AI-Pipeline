import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/dataset.csv")
X = df["text"]
y = df["label"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

with open("ml/model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

