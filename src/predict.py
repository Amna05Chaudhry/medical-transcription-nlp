"""Inference + word-level explanations for the TF-IDF + logistic-regression model.

For a linear model the logit of class k is  sum_j tfidf_j(x) * W[k, j] + b_k,
so tfidf_j(x) * W[k, j] is exactly how much term j pushed the note towards
class k. We return the strongest positive terms for the predicted class and
use them to highlight the original text.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .data import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tfidf_logreg.joblib"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train it with: python -m src.train_linear")
    return joblib.load(MODEL_PATH)


def predict(text: str, top_k_terms: int = 12) -> dict:
    pipe = load_model()
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    cleaned = clean_text(text)
    x = vec.transform([cleaned])
    proba = clf.predict_proba(x)[0]
    classes = list(clf.classes_)
    order = np.argsort(proba)[::-1]
    k = order[0]

    terms = vec.get_feature_names_out()
    cols = x.indices
    contrib = x.data * clf.coef_[k, cols]
    pos = np.argsort(contrib)[::-1]
    keywords = [(terms[cols[i]], float(contrib[i])) for i in pos[:top_k_terms] if contrib[i] > 0]
    against = [(terms[cols[i]], float(contrib[i])) for i in pos[::-1][:5] if contrib[i] < 0]
    return {
        "label": classes[k],
        "confidence": float(proba[k]),
        "probs": {classes[i]: float(proba[i]) for i in order},
        "keywords": keywords,
        "against": against,
        "n_known_terms": int(len(cols)),
    }


def highlight(text: str, keywords: list[tuple[str, float]]) -> list[tuple[str, str | None]]:
    """Split `text` into (span, tag) pieces for gr.HighlightedText.

    Bigram keywords are matched across any run of non-letters, unigrams as
    whole words, case-insensitively, so they line up with the raw note.
    """
    if not keywords:
        return [(text, None)]
    weights = {kw: w for kw, w in keywords}
    pats = sorted(weights, key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(r"[^a-zA-Z]+".join(map(re.escape, p.split())) for p in pats) + r")\b",
                    re.IGNORECASE)
    out, last = [], 0
    top = max(weights.values())
    for m in rx.finditer(text):
        key = re.sub(r"[^a-z]+", " ", m.group(0).lower()).strip()
        w = weights.get(key)
        if w is None:
            continue
        if m.start() > last:
            out.append((text[last:m.start()], None))
        out.append((m.group(0), "strong" if w >= 0.5 * top else "supporting"))
        last = m.end()
    out.append((text[last:], None))
    return out
