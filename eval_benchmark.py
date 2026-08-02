#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_benchmark.py

Evaluate a trained token-classification model on the REAL hand-annotated
benchmark (benchmark_full.csv). This is the honest test: the benchmark is real
news text the model never saw, annotated by hand to your rules. Compare these
numbers to your synthetic-influenced test set to see true generalization.

It:
  1. loads the model + tokenizer from --model-dir
  2. tokenizes each benchmark sentence, runs prediction
  3. converts char-offset gold entities -> token BIO aligned the same way
  4. reports entity-level precision/recall/F1 overall + per label (seqeval)
  5. optionally dumps per-sentence errors (--errors errors.txt) so you can see
     exactly where the model fails (great for the next data-fix iteration).

Usage:
    python eval_benchmark.py --model-dir model_out --benchmark benchmark_full.csv
    python eval_benchmark.py --model-dir model_out --benchmark benchmark_full.csv \
        --errors benchmark_errors.txt
"""
import csv, json, ast, argparse, re
import numpy as np

SCHEMA=["PERSON","ORGANIZATION","LOCATION","JOB","PRODUCT","WORKOFART","TIMEDATE","AMOUNT"]

def parse_cell(v):
    if isinstance(v,(list,dict)): return v
    if not isinstance(v,str) or not v.strip(): return []
    try: return json.loads(v)
    except json.JSONDecodeError: return ast.literal_eval(v)

_tok_re=re.compile(r"\S+")
def tokenize_offsets(text):
    return [(m.group(0),m.start(),m.end()) for m in _tok_re.finditer(text)]

def gold_bio(text, entities):
    toks=tokenize_offsets(text)
    tags=["O"]*len(toks)
    for e in sorted(entities,key=lambda x:(x["start"],-(x["end"]-x["start"]))):
        started=False
        for i,(tk,s,en) in enumerate(toks):
            if s<e["end"] and e["start"]<en:
                if tags[i]=="O":
                    tags[i]=("B-" if not started else "I-")+e["label"]
                    started=True
    return [t[0] for t in toks], tags

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",required=True)
    ap.add_argument("--benchmark",required=True)
    ap.add_argument("--errors")
    ap.add_argument("--max-length",type=int,default=256)
    a=ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import evaluate
    seqeval=evaluate.load("seqeval")

    tok=AutoTokenizer.from_pretrained(a.model_dir)
    model=AutoModelForTokenClassification.from_pretrained(a.model_dir)
    model.eval()
    id2label=model.config.id2label

    rows=list(csv.DictReader(open(a.benchmark,encoding="utf-8")))
    all_true=[]; all_pred=[]; error_lines=[]

    for r in rows:
        text=r["text"]; ents=parse_cell(r["entities"])
        words, gold = gold_bio(text, ents)
        if not words: continue
        enc=tok(words, is_split_into_words=True, return_tensors="pt",
                truncation=True, max_length=a.max_length)
        with torch.no_grad():
            logits=model(**enc).logits
        preds=logits.argmax(-1)[0].tolist()
        word_ids=enc.word_ids(0)
        pred_tags=["O"]*len(words); seen=set()
        for idx,wid in enumerate(word_ids):
            if wid is None or wid in seen: continue
            seen.add(wid)
            pred_tags[wid]=id2label[preds[idx]]
        all_true.append(gold); all_pred.append(pred_tags)
        # record errors
        for w,g,p in zip(words,gold,pred_tags):
            if g!=p:
                error_lines.append(f"[{r['id']}] {w!r}  gold={g}  pred={p}")

    res=seqeval.compute(predictions=all_pred, references=all_true, zero_division=0)
    print("="*56); print("BENCHMARK EVALUATION (real, hand-annotated)"); print("="*56)
    print(f"  sentences      : {len(all_true)}")
    print(f"  overall precision: {res['overall_precision']:.4f}")
    print(f"  overall recall   : {res['overall_recall']:.4f}")
    print(f"  overall F1       : {res['overall_f1']:.4f}")
    print(f"  overall accuracy : {res['overall_accuracy']:.4f}")
    print("\n  Per-label:")
    print(f"    {'label':14}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
    for lab in SCHEMA:
        if lab in res:
            d=res[lab]
            print(f"    {lab:14}{d['precision']:8.3f}{d['recall']:8.3f}{d['f1']:8.3f}{d['number']:9}")
        else:
            print(f"    {lab:14}{'--':>8}{'--':>8}{'--':>8}{0:9}")

    if a.errors:
        open(a.errors,"w",encoding="utf-8").write("\n".join(error_lines))
        print(f"\n  Wrote {len(error_lines)} token errors -> {a.errors}")

if __name__=="__main__":
    main()
