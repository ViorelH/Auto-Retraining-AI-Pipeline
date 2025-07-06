import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/dataset.csv")
X = df["text"]
y_true = df["label"]

with open("ml/model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

acc = accuracy_score(y_true, y_pred)
with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {acc}\n")
