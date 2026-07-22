#!/usr/bin/env python3
"""
prepare_ft_data_v2.py — improved fine-tuning data for MedReason.

Fixes the open-ended regression from v1:
  * Open-ended TARGET answer = the gold `answer` (what GT scores), NOT the caption.
  * Reasoning trace is a short lead-in derived from the gold answer + visual_description,
    framed to support the answer (not a standalone caption).
  * UPWEIGHT open-ended by duplicating it `--open-weight` times so the 81/19
    MCQ/open imbalance does not crush open-ended learning.

Usage:
  python prepare_ft_data_v2.py \
      --json data/train/medreason_train_selection.json \
      --imgs data/train/imgs --out ft_data_v2 \
      --val-n 1000 --open-weight 3
"""
import argparse, json, os, random
from collections import Counter

MCQ_SYSTEM = ("You are an expert radiologist answering a multiple-choice question about a "
              "medical image. Examine the image carefully, then choose the single best option.")
OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. Examine the image carefully and answer with clinical precision.")


def mcq_prompt(case):
    opts = "\n".join(f"{k}. {case[k]}" for k in ("A", "B", "C", "D", "E") if k in case)
    return (f"{MCQ_SYSTEM}\n\nQuestion: {case['question']}\n\nOptions:\n{opts}\n\n"
            "Answer with ONLY the single letter of the best option (A, B, C, D, or E). "
            "Output just the letter, nothing else.")


def open_prompt(case):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {case['question']}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image.\n'
            '- "answer": a single specific, committed clinical statement.')


def build(case, imgs_dir):
    img = os.path.join(imgs_dir, os.path.basename(case["image_path"]))
    if not os.path.exists(img):
        return None
    qt = case.get("question type")
    if qt == "mcq":
        gold = str(case.get("answer", "")).strip().upper()
        if gold not in {"A", "B", "C", "D", "E"}:
            return None
        return {"image": img, "prompt": mcq_prompt(case), "target": gold,
                "task_type": "mcq", "case_id": case["case_id"]}
    else:
        answer = str(case.get("answer", "")).strip()
        if not answer:
            return None
        vd = str(case.get("visual_description", "")).strip()
        # Reasoning trace SUPPORTS the gold answer; we lead with visible evidence
        # (from the caption if present) then point to the answer. The key change:
        # the "answer" field is the GOLD answer, which is what GT scores.
        if vd:
            trace = f"{vd} These findings support the conclusion."
        else:
            trace = f"The visible findings support the following conclusion."
        target = json.dumps({"reasoning_trace": trace, "answer": answer}, ensure_ascii=False)
        return {"image": img, "prompt": open_prompt(case), "target": target,
                "task_type": "open", "case_id": case["case_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--open-weight", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = json.load(open(args.json))["cases"]
    ex = [e for e in (build(c, args.imgs) for c in cases) if e]
    random.seed(args.seed); random.shuffle(ex)

    val = ex[:args.val_n]
    train = ex[args.val_n:]

    # upweight open-ended in TRAIN only (val stays single-copy for honest eval)
    weighted = []
    for e in train:
        weighted.append(e)
        if e["task_type"] == "open":
            for _ in range(args.open_weight - 1):
                weighted.append(e)
    random.shuffle(weighted)

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(weighted, os.path.join(args.out, "train.jsonl"))
    dump(val, os.path.join(args.out, "val.jsonl"))
    print(f"train (after {args.open_weight}x open upweight): {len(weighted)} "
          f"({Counter(e['task_type'] for e in weighted)})")
    print(f"val: {len(val)} ({Counter(e['task_type'] for e in val)})")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
