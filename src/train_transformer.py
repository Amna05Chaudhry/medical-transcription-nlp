"""Small transformer fine-tune on CPU (DistilBERT by default).

Notes are long (median ~400 words) but we keep the first MAX_TOKENS word-pieces
to keep CPU training to a few minutes per epoch. Model selection uses val
macro-F1; the test split is scored once. Weights (~250 MB) are too big to
commit, so only the metrics are saved; the app serves the linear model.

    python -m src.train_transformer                      # distilbert-base-uncased
    python -m src.train_transformer --model emilyalsentzer/Bio_ClinicalBERT
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import load_splits
from .metrics import evaluate, fmt, save_result

torch.set_num_threads(8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()
    torch.manual_seed(42)

    train, val, test = load_splits()
    classes = sorted(train["label"].unique())
    cidx = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(classes))

    def enc(df):
        # raw transcription keeps casing/numbers the tokenizer knows how to use
        e = tok(df["transcription"].tolist(), truncation=True, max_length=args.max_tokens,
                padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    (tr_ids, tr_m), (va_ids, va_m), (te_ids, te_m) = enc(train), enc(val), enc(test)
    ytr = torch.tensor([cidx[c] for c in train["label"]])
    counts = np.bincount(ytr.numpy(), minlength=len(classes))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (len(classes) * counts), dtype=torch.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(ytr) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    def predict(ids, mask):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(input_ids=ids[i:i + 64], attention_mask=mask[i:i + 64]).logits
                                   .argmax(-1).numpy() for i in range(0, len(ids), 64)])

    best_f1, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = loss_fn(model(input_ids=tr_ids[b], attention_mask=tr_m[b]).logits, ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        vm = evaluate(val["label"], [classes[i] for i in predict(va_ids, va_m)], classes)
        history.append({"epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"],
                        "elapsed_s": round(time.time() - t0)})
        print(f"epoch {epoch} val {fmt(vm)} ({time.time() - t0:.0f}s)", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    secs = time.time() - t0
    model.load_state_dict(best_state)
    val_m = evaluate(val["label"], [classes[i] for i in predict(va_ids, va_m)], classes)
    test_m = evaluate(test["label"], [classes[i] for i in predict(te_ids, te_m)], classes)
    short = args.model.split("/")[-1]
    name = f"{short} fine-tune ({args.max_tokens} tok)"
    print(f"{name}: val {fmt(val_m)} | test {fmt(test_m)} | {secs:.0f}s")
    save_result(name, val_m, test_m, {"history": history, "train_seconds": round(secs, 1), "classes": classes,
                                      "params": vars(args)})


if __name__ == "__main__":
    main()
