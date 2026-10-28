import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=3))
    print("Confusion matrix (rows and columns: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    if name == "Naive Bayes":
        joblib.dump(model, "spam_model.joblib")
    else:
        joblib.dump(model, "logistic_regression_model.joblib")