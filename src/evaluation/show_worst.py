#!/usr/bin/env python3
"""
show_worst.py — ERROR ANALYSIS. No GPU needed.

Prints the worst-scoring cases side by side (question / reference / model answer)
so you can SEE how the open-ended answers actually fail. Also breaks scores down
by question type, because the aggregate hides which categories are near zero.

We have never done this. Every fix so far has been a guess at the failure mode;
an hour reading these will say more than another GPU run.

Usage (login node is fine):
  python show_worst.py --judged judged_base.jsonl --worst 30
  python show_worst.py --judged judged_base.jsonl --compare judged_loraopen.jsonl
"""
import argparse, json, re
from collections import defaultdict
import statistics


def qtype(question):
    """Coarse question-type buckets for stratified analysis."""
    q = question.lower()
    if "modality" in q or "what type of imaging" in q or "what kind of scan" in q:
        return "modality"
    if "diagnos" in q or "most likely" in q or "condition" in q or "disease" in q:
        return "diagnosis"
    if "describe" in q or "characteristic" in q or "appearance" in q or "pattern" in q:
        return "description"
    if "where" in q or "location" in q or "which structure" in q or "anatom" in q:
        return "location"
    if "cause" in q or "etiolog" in q or "why" in q:
        return "etiology"
    if "treat" in q or "management" in q or "next step" in q:
        return "management"
    return "other"


def load(path):
    return [json.loads(l) for l in open(path)]


def summarise(rows, label):
    scores = [r["gt_score"] for r in rows]
    print(f"\n{label}: mean GT {statistics.mean(scores):.3f} over {len(scores)} cases")
    by = defaultdict(list)
    for r in rows:
        by[qtype(r["question"])].append(r["gt_score"])
    print(f"  {'question type':<14}{'n':>5}{'mean GT':>10}")
    for k in sorted(by, key=lambda k: statistics.mean(by[k])):
        print(f"  {k:<14}{len(by[k]):>5}{statistics.mean(by[k]):>10.3f}")
    # length vs score — is the model penalised for being vague or for being long?
    lo = [r["answer_words"] for r in rows if r["gt_score"] <= 1]
    hi = [r["answer_words"] for r in rows if r["gt_score"] >= 3]
    if lo and hi:
        print(f"  mean answer length  low-scoring (<=1): {statistics.mean(lo):.1f} words")
        print(f"  mean answer length high-scoring (>=3): {statistics.mean(hi):.1f} words")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--compare", default=None,
                    help="second judged file to compare against")
    ap.add_argument("--worst", type=int, default=30)
    args = ap.parse_args()

    rows = load(args.judged)
    summarise(rows, f"A: {args.judged}")

    if args.compare:
        rows_b = load(args.compare)
        summarise(rows_b, f"B: {args.compare}")
        bid = {r["case_id"]: r for r in rows_b}
        deltas = [(r["gt_score"] - bid[r["case_id"]]["gt_score"], r)
                  for r in rows if r["case_id"] in bid]
        better = sum(1 for d, _ in deltas if d < 0)
        worse = sum(1 for d, _ in deltas if d > 0)
        print(f"\n  head-to-head on {len(deltas)} shared cases: "
              f"B better on {better}, A better on {worse}, "
              f"tied on {len(deltas)-better-worse}")

    print("\n" + "=" * 78)
    print(f"WORST {args.worst} CASES — read these")
    print("=" * 78)
    for r in sorted(rows, key=lambda r: r["gt_score"])[:args.worst]:
        print(f"\n[{r['case_id']}]  GT={r['gt_score']}  type={qtype(r['question'])}")
        print(f"  Q     : {r['question'][:220]}")
        print(f"  GOLD  : {r['gold'][:220]}")
        print(f"  MODEL : {r['pred_answer'][:220]}")
        if r.get("judge_raw"):
            reason = re.sub(r'\s+', ' ', r["judge_raw"])
            print(f"  JUDGE : {reason[:200]}")

    print("\n" + "=" * 78)
    print("Questions to answer while reading:")
    print("  1. Wrong anatomy/modality, or right area but not specific enough?")
    print("  2. Is the model hedging where the reference commits?")
    print("  3. Is it answering a different question than the one asked?")
    print("  4. Which question type is worst — is that where the points are?")


if __name__ == "__main__":
    main()
