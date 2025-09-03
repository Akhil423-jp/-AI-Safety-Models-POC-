
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

try:
    from .utils import normalize_text
except Exception:
    from utils import normalize_text

def main():
    import os
    (Path(__file__).resolve().parents[1] / 'models').mkdir(exist_ok=True)
    print('DEBUG base=', Path(__file__).resolve().parents[1])
    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(base / "data" / "abuse_toy.csv")
    df["text"] = df["text"].astype(str).apply(normalize_text)

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    joblib.dump(pipe, base / "models" / "abuse_model.joblib")
    (base / "models" / "abuse_metrics.json").write_text(pd.Series(report).to_json())

if __name__ == "__main__":
    main()
