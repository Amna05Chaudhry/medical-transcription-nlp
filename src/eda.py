"""Pre-compute everything the EDA tab shows, so the app starts instantly.

Writes results/eda.json with:
* raw vs cleaned class counts,
* top TF-IDF terms per specialty (training split only),
* a 2-D t-SNE of every note (TF-IDF -> 50-d truncated SVD -> t-SNE).
  The original notebook ran t-SNE on 100 random words from pretrained
  word2vec/GloVe/FastText; projecting whole documents shows whether the
  specialties actually separate, which is what this project is about.

    python -m src.eda
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from .data import RAW_CSV, ROOT, load_clean, raw_stats, split
from .metrics import RESULTS_DIR


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    clean = load_clean()
    train, val, test = split(clean)
    split_name = pd.Series("train", index=clean.index)
    key = clean["transcription"]
    split_name[key.isin(val["transcription"])] = "val"
    split_name[key.isin(test["transcription"])] = "test"

    vec = TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform(train["text"])
    terms = np.array(vec.get_feature_names_out())
    top_terms = {}
    for lab in sorted(train["label"].unique()):
        mean = np.asarray(Xtr[(train["label"] == lab).to_numpy()].mean(axis=0)).ravel()
        order = mean.argsort()[::-1][:15]
        top_terms[lab] = [[terms[i], round(float(mean[i]), 4)] for i in order]

    X = vec.transform(clean["text"])
    Z = TruncatedSVD(50, random_state=0).fit_transform(X)
    xy = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)

    out = {
        "stats": raw_stats(),
        "raw_counts": raw["medical_specialty"].str.strip().value_counts().to_dict(),
        "clean_counts": clean["label"].value_counts().to_dict(),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "top_terms": top_terms,
        "tsne": {
            "x": np.round(xy[:, 0], 2).tolist(),
            "y": np.round(xy[:, 1], 2).tolist(),
            "label": clean["label"].tolist(),
            "title": clean["sample_name"].fillna("").str.strip().tolist(),
            "split": split_name.tolist(),
        },
        "word_counts": clean["text"].str.split().str.len().describe().round(1).to_dict(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eda.json").write_text(json.dumps(out))
    write_examples(test)
    print(out["stats"], out["split_sizes"])


EXAMPLE_SPECIALTIES = ["Cardiovascular / Pulmonary", "Orthopedic", "Gastroenterology", "Obstetrics / Gynecology",
                       "Ophthalmology"]


def write_examples(test: pd.DataFrame) -> None:
    """Save one held-out test note per specialty to examples/ for the app's sample picker."""
    ex_dir = ROOT / "examples"
    ex_dir.mkdir(exist_ok=True)
    for spec in EXAMPLE_SPECIALTIES:
        rows = test[test["label"] == spec]
        rows = rows[rows["transcription"].str.len().between(800, 4000)]
        row = rows.iloc[0]
        fname = re.sub(r"[^A-Za-z0-9]+", "_", f"{spec.split(' ')[0]} {row['sample_name'].strip()}").strip("_")
        body = row["transcription"].strip()
        body = re.sub(r",(?=[A-Z][A-Z /&-]{2,}:)", "\n\n", body)  # ",HISTORY:" -> new paragraph
        body = re.sub(r":\s*,\s*", ": ", body)                     # "DIAGNOSIS:,  1." -> "DIAGNOSIS: 1."
        body = re.sub(r"\.,\s*(?=\d+\.)", ".\n", body)              # ".,2." list items -> new line
        body = re.sub(r"[ \t]{2,}", " ", body)
        (ex_dir / f"{fname[:60]}.txt").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
