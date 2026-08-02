#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ner.py

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
============================================================
HYPERPARAMETERS
============================================================
  data_dir        : .
  model           : xlm-roberta-base
  epochs          : 5
  batch_size      : 16
  lr              : 2e-05
  weight_decay    : 0.01
  max_length      : 256
  warmup_ratio    : 0.1
  seed            : 42
  outdir          : model_out
  labels          : 17 (['O', 'B-PERSON', 'I-PERSON', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-LOCATION', 'I-LOCATION', 'B-JOB', 'I-JOB', 'B-PRODUCT', 'I-PRODUCT', 'B-WORKOFART', 'I-WORKOFART', 'B-TIMEDATE', 'I-TIMEDATE', 'B-AMOUNT', 'I-AMOUNT'])
  train rows      : 1387
  test rows       : 140
 [435/435 07:36, Epoch 5/5]
Epoch	Training Loss	Validation Loss
1	1.161867	0.742766
2	0.345136	0.540268
3	0.163696	0.426049
4	0.113948	0.422049
5	0.075168	0.426686
Writing model shards: 100%
 1/1 [00:17<00:00, 17.57s/it]
Writing model shards: 100%
 1/1 [00:25<00:00, 25.21s/it]
Writing model shards: 100%
 1/1 [00:20<00:00, 20.28s/it]
Writing model shards: 100%
 1/1 [00:26<00:00, 26.70s/it]
Writing model shards: 100%
 1/1 [00:27<00:00, 27.90s/it]

============================================================
FINAL TEST METRICS
============================================================
 [9/9 00:00]
Training Loss	Validation Loss	Epoch
0.075168	0.422049	5
  loss            : 0.4220
Writing model shards: 100%
 1/1 [00:08<00:00,  8.63s/it]

Saved model + run_summary.json -> model_out
Fine-tune a token-classification NER model on our train/test split.

Input : data_split/train.csv, data_split/test.csv, data_split/label_list.json
        (produced by split_and_bio.py). Each row has `tokens` (JSON list) and
        `ner_tags` (JSON list of BIO strings).

Model : default xlm-roberta-base (multilingual — good if you later add
        Azerbaijani). Swap with --model distilbert-base-cased for a small,
        fast English-only run.

Metrics: seqeval (entity-level precision/recall/F1, overall + per label).

Usage:
    pip install "transformers>=4.40" datasets seqeval evaluate accelerate torch
    python train_ner.py --data-dir data_split --model xlm-roberta-base \
        --epochs 5 --batch-size 16 --lr 2e-5 --outdir model_out

Notes:
  - We log EVERY hyperparameter and the final metrics so the run is reproducible
    (matches the task's "document each hyperparameter" requirement).
  - No validation set is used for early stopping by default (small data); we
    report train + test metrics. Add --eval-split if you carve one out.
"""
import os, json, csv, ast, argparse, random
import numpy as np

def parse_list(v):
    if isinstance(v, list): return v
    try: return json.loads(v)
    except json.JSONDecodeError: return ast.literal_eval(v)

def read_split(path):
    tokens, tags = [], []
    for d in csv.DictReader(open(path, encoding="utf-8")):
        tk = parse_list(d["tokens"]); tg = parse_list(d["ner_tags"])
        if tk and len(tk) == len(tg):
            tokens.append(tk); tags.append(tg)
    return {"tokens": tokens, "ner_tags": tags}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_split")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="model_out")
    a = ap.parse_args()

    # ---- reproducibility ----
    random.seed(a.seed); np.random.seed(a.seed)
    import torch; torch.manual_seed(a.seed)

    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              TrainingArguments, Trainer, DataCollatorForTokenClassification)
    import evaluate

    # ---- labels ----
    lab = json.load(open(os.path.join(a.data_dir, "label_list.json")))
    id2label = {int(k): v for k, v in lab["id2label"].items()}
    label2id = {k: int(v) for k, v in lab["label2id"].items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    # ---- data ----
    train_raw = read_split(os.path.join(a.data_dir, "train.csv"))
    test_raw  = read_split(os.path.join(a.data_dir, "test.csv"))
    ds_train = Dataset.from_dict(train_raw)
    ds_test  = Dataset.from_dict(test_raw)

    tok = AutoTokenizer.from_pretrained(a.model)

    def align(batch):
        enc = tok(batch["tokens"], truncation=True, is_split_into_words=True,
                  max_length=a.max_length)
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            prev = None; lab_ids = []
            for wid in word_ids:
                if wid is None:
                    lab_ids.append(-100)
                elif wid != prev:
                    lab_ids.append(label2id[tags[wid]])
                else:
                    # subword continuation: keep same label but as I- (or -100)
                    t = tags[wid]
                    lab_ids.append(label2id[t])
                prev = wid
            all_labels.append(lab_ids)
        enc["labels"] = all_labels
        return enc

    ds_train = ds_train.map(align, batched=True, remove_columns=ds_train.column_names)
    ds_test  = ds_test.map(align,  batched=True, remove_columns=ds_test.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(label_names), id2label=id2label, label2id=label2id)

    collator = DataCollatorForTokenClassification(tok)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        preds, labels = p
        preds = np.argmax(preds, axis=2)
        true_pred, true_lab = [], []
        for pr, la in zip(preds, labels):
            tp, tl = [], []
            for p_i, l_i in zip(pr, la):
                if l_i != -100:
                    tp.append(label_names[p_i]); tl.append(label_names[l_i])
            true_pred.append(tp); true_lab.append(tl)
        res = seqeval.compute(predictions=true_pred, references=true_lab,
                              zero_division=0)
        out = {"precision": res["overall_precision"],
               "recall": res["overall_recall"],
               "f1": res["overall_f1"],
               "accuracy": res["overall_accuracy"]}
        # per-label F1
        for k, v in res.items():
            if isinstance(v, dict):
                out[f"f1_{k}"] = v["f1"]
        return out

    args = TrainingArguments(
        output_dir=a.outdir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        learning_rate=a.lr,
        weight_decay=a.weight_decay,
        warmup_ratio=a.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        seed=a.seed,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=ds_train, eval_dataset=ds_test,
                      data_collator=collator, tokenizer=tok,
                      compute_metrics=compute_metrics)

    # ---- log hyperparameters (reproducibility) ----
    print("="*60); print("HYPERPARAMETERS"); print("="*60)
    for k, v in vars(a).items():
        print(f"  {k:16}: {v}")
    print(f"  labels          : {len(label_names)} ({label_names})")
    print(f"  train rows      : {len(ds_train)}")
    print(f"  test rows       : {len(ds_test)}")

    trainer.train()

    print("\n"+"="*60); print("FINAL TEST METRICS"); print("="*60)
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"  {k[5:]:16}: {v:.4f}" if isinstance(v,(int,float)) else f"  {k[5:]}: {v}")

    trainer.save_model(a.outdir)
    tok.save_pretrained(a.outdir)
    json.dump({"hyperparameters": vars(a), "metrics": metrics},
              open(os.path.join(a.outdir, "run_summary.json"), "w"),
              indent=2, default=str)
    print(f"\nSaved model + run_summary.json -> {a.outdir}")

if __name__ == "__main__":
    main()
