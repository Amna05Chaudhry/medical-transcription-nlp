"""LSTM and GRU classifiers, a corrected port of the original notebook.

The notebook's Keras models had a single sigmoid unit + binary cross-entropy on
a ~40-class integer label, which is why its loss went to -3000 and accuracy sat
at 0.15%. They also trained on the short `description` field. Here each model:

* reads the cleaned `transcription` (first MAX_LEN tokens),
* uses a 128-d embedding + a bidirectional LSTM / GRU (128 units per direction),
* max-pools over time, then a linear layer with one logit per specialty,
* trains with class-weighted cross-entropy and early stopping on val macro-F1.

    python -m src.train_rnn
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import load_splits
from .metrics import evaluate, fmt, save_result

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MAX_LEN = 300
VOCAB_SIZE = 20_000
EPOCHS = 12
PATIENCE = 3
BATCH = 32
SEED = 42

torch.set_num_threads(4)


def build_vocab(texts) -> dict[str, int]:
    counts = collections.Counter(w for t in texts for w in t.split())
    itos = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 2)]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, vocab) -> torch.Tensor:
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()[:MAX_LEN]]
        out[i, : len(ids)] = ids
    return torch.from_numpy(out)


class RNNClassifier(nn.Module):
    def __init__(self, cell: str, vocab_size: int, n_classes: int, emb: int = 128, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.drop(self.emb(x)))
        h = h.masked_fill(~mask, -1e4).max(dim=1).values  # max-pool over real tokens
        return self.out(self.drop(h))


def predict(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 128]).argmax(-1) for i in range(0, len(X), 128)]).numpy()


def train_one(cell: str, train, val, test, classes, vocab) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cidx = {c: i for i, c in enumerate(classes)}
    Xtr, Xva, Xte = (encode(d["text"].tolist(), vocab) for d in (train, val, test))
    ytr = torch.tensor([cidx[c] for c in train["label"]])

    counts = np.bincount(ytr.numpy(), minlength=len(classes))
    weights = torch.tensor(len(ytr) / (len(classes) * counts), dtype=torch.float32)
    model = RNNClassifier(cell, len(vocab), len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        vm = evaluate(val["label"], [classes[i] for i in predict(model, Xva)], classes)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_macro_f1": vm["macro_f1"],
                        "val_accuracy": vm["accuracy"]})
        print(f"  {cell} epoch {epoch:2d} loss {total / len(Xtr):.3f} val {fmt(vm)} ({time.time() - t0:.0f}s)", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_state, bad = vm["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    secs = time.time() - t0
    model.load_state_dict(best_state)
    val_m = evaluate(val["label"], [classes[i] for i in predict(model, Xva)], classes)
    test_m = evaluate(test["label"], [classes[i] for i in predict(model, Xte)], classes)
    name = f"Bi{cell.upper()} (PyTorch)"
    print(f"{name}: val {fmt(val_m)} | test {fmt(test_m)} | {secs:.0f}s")
    save_result(name, val_m, test_m, {"history": history, "train_seconds": round(secs, 1), "classes": classes})
    torch.save(best_state, MODELS_DIR / f"{cell}.pt")


def main() -> None:
    train, val, test = load_splits()
    classes = sorted(train["label"].unique())
    MODELS_DIR.mkdir(exist_ok=True)
    vocab = build_vocab(train["text"])
    (MODELS_DIR / "rnn_vocab.json").write_text(json.dumps({"vocab": vocab, "classes": classes}))
    for cell in ("lstm", "gru"):
        train_one(cell, train, val, test, classes, vocab)


if __name__ == "__main__":
    main()
