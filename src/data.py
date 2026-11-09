"""Load MTSamples, clean the labels, and build a leak-free train/val/test split.

MTSamples has two traps that a naive `train_test_split` falls into:

1. **Duplicates.** About half the rows are copies: the same transcription is
   filed once under a clinical specialty (e.g. "Orthopedic") and again under a
   *document-type* bucket (e.g. "Surgery", "Consult - History and Phy.").
   A random split puts copies on both sides, which leaks test text into
   training and gives the same text two contradictory labels.
2. **Document types are not specialties.** "Surgery", "SOAP / Chart / Progress
   Notes", "Discharge Summary", ... describe the *kind of note*, not the
   specialty, and they swallow notes from every specialty.

So we: drop empty texts, drop the document-type buckets, collapse duplicates to
one row per unique text, drop texts that are still tagged with more than one
clinical specialty (ambiguous), keep specialties with >= MIN_SAMPLES unique
notes, then do a stratified 70/15/15 split.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "mtsamples.csv"
SEED = 42
MIN_SAMPLES = 40

DOC_TYPE_LABELS = {
    "Surgery",
    "Consult - History and Phy.",
    "SOAP / Chart / Progress Notes",
    "Discharge Summary",
    "Office Notes",
    "Letters",
    "Emergency Room Reports",
    "IME-QME-Work Comp etc.",
}


def clean_text(text: str) -> str:
    """Light normalisation shared by every model (and the app).

    MTSamples uses ",SECTION:," as a section separator; we turn those commas
    back into spaces. We keep words un-stemmed so that the keywords the linear
    model highlights can be found again in the original note.
    """
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)          # stray HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r",(?=[A-Z ]{3,}:)", " ", text)  # ",HISTORY:" -> " HISTORY:"
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only
    return re.sub(r"\s+", " ", text).strip()


def load_clean(min_samples: int = MIN_SAMPLES, csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Return one row per unique transcription with a single specialty label."""
    df = pd.read_csv(csv_path)
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df = df.dropna(subset=["transcription"])
    df = df[df["transcription"].str.strip().str.len() > 0]
    df = df[~df["medical_specialty"].isin(DOC_TYPE_LABELS)]

    grouped = (
        df.groupby("transcription")
        .agg(labels=("medical_specialty", lambda s: sorted(set(s))),
             description=("description", "first"),
             sample_name=("sample_name", "first"))
        .reset_index()
    )
    grouped = grouped[grouped["labels"].str.len() == 1].copy()
    grouped["label"] = grouped["labels"].str[0]

    counts = grouped["label"].value_counts()
    keep = counts[counts >= min_samples].index
    out = grouped[grouped["label"].isin(keep)].drop(columns="labels")
    out["text"] = out["transcription"].map(clean_text)
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = SEED):
    """Stratified 70 / 15 / 15 split."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits():
    return split(load_clean())


def raw_stats(csv_path: Path = RAW_CSV) -> dict:
    """Numbers for the README / EDA tab: how much the cleaning removes."""
    df = pd.read_csv(csv_path)
    clean = load_clean(csv_path=csv_path)
    return {
        "raw_rows": int(len(df)),
        "raw_specialties": int(df["medical_specialty"].nunique()),
        "raw_unique_texts": int(df["transcription"].dropna().nunique()),
        "clean_rows": int(len(clean)),
        "clean_classes": int(clean["label"].nunique()),
    }


if __name__ == "__main__":
    tr, va, te = load_splits()
    print(raw_stats())
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    print(pd.concat([tr, va, te])["label"].value_counts().to_string())
