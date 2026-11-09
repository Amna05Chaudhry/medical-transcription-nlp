"""TF-IDF + linear classifiers (the strong, explainable baseline).

Hyper-parameters are chosen on the validation split only; the test split is
scored once at the end. The best pipeline is saved to models/tfidf_logreg.joblib
and is the model the app uses (its coefficients give per-word explanations).

    python -m src.train_linear
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import load_splits
from .metrics import evaluate, fmt, save_result

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                           stop_words="english", max_features=50_000)


CANDIDATES = {
    "TF-IDF + Logistic Regression": lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
    "TF-IDF + Linear SVM": lambda C: LinearSVC(C=C, class_weight="balanced"),
    "TF-IDF + Complement NB": lambda C: ComplementNB(alpha=C),
}
GRID = {
    "TF-IDF + Logistic Regression": [1, 3, 10, 30, 100],
    "TF-IDF + Linear SVM": [0.1, 0.3, 1, 3],
    "TF-IDF + Complement NB": [0.01, 0.03, 0.1, 0.3, 1.0],
}


def main() -> None:
    train, val, test = load_splits()
    classes = sorted(train["label"].unique())
    MODELS_DIR.mkdir(exist_ok=True)

    for name, make in CANDIDATES.items():
        best = None
        for C in GRID[name]:
            pipe = Pipeline([("tfidf", tfidf()), ("clf", make(C))])
            pipe.fit(train["text"], train["label"])
            m = evaluate(val["label"], pipe.predict(val["text"]), classes)
            if best is None or m["macro_f1"] > best[1]["macro_f1"]:
                best = (C, m, pipe)
        C, val_m, _ = best
        # refit on train only with the chosen C, then score test once
        t0 = time.time()
        pipe = Pipeline([("tfidf", tfidf()), ("clf", make(C))])
        pipe.fit(train["text"], train["label"])
        fit_s = time.time() - t0
        test_m = evaluate(test["label"], pipe.predict(test["text"]), classes)
        print(f"{name:32s} C={C:<6} val {fmt(val_m)} | test {fmt(test_m)} | fit {fit_s:.1f}s")
        save_result(name, val_m, test_m, {"params": {"C_or_alpha": C}, "train_seconds": round(fit_s, 1),
                                          "classes": classes})
        if name == "TF-IDF + Logistic Regression":
            joblib.dump(pipe, MODELS_DIR / "tfidf_logreg.joblib", compress=3)


if __name__ == "__main__":
    main()
