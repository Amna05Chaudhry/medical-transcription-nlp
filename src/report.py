"""Print the results table (markdown) from results/results.json, best first.

    python -m src.report
"""
from __future__ import annotations

import json

from .metrics import RESULTS_JSON


def main() -> None:
    res = json.loads(RESULTS_JSON.read_text())
    print("| Model | Test macro-F1 | Test accuracy | Val macro-F1 | Train time (CPU) |")
    print("|---|---:|---:|---:|---:|")
    for name in sorted(res, key=lambda k: -res[k]["test"]["macro_f1"]):
        r = res[name]
        print(f"| {name} | {r['test']['macro_f1']:.3f} | {r['test']['accuracy']:.3f} | "
              f"{r['val']['macro_f1']:.3f} | {r.get('train_seconds', 0):.0f} s |")


if __name__ == "__main__":
    main()
