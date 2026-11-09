"""Shared evaluation: macro-F1 / accuracy, per-class report, confusion matrix, results.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"


def evaluate(y_true, y_pred, labels) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_result(name: str, val: dict, test: dict, extra: dict | None = None) -> None:
    """Merge one model's numbers into results/results.json (one key per model)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    data = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    data[name] = {"val": val, "test": test, **(extra or {})}
    RESULTS_JSON.write_text(json.dumps(data, indent=2))


def fmt(m: dict) -> str:
    return f"macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}"


def to_index(labels_list, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[c] for c in labels_list])
