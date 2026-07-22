#!/usr/bin/env python3
"""
prepare_ft_data.py — build LoRA fine-tuning data for MedReason from the train split.

Converts each train case into a Qwen2.5-VL chat example:
  user: [image] + task prompt   ->   assistant: gold target
Uses the SAME prompts as inference so train and test match.

MCQ target   : the gold option letter (matches bare-letter inference).
Open target  : JSON {"reasoning_trace": <visual_description>, "answer": <gold answer>}
               (we use the train 'visual_description' as a grounded trace target,
                and the gold 'answer' as the final answer.)

Outputs JSONL: train.jsonl + val.jsonl, each line:
  {"image": "<abs path>", "prompt": "<user text>", "target": "<assistant text>",
   "task_type": "mcq"|"open", "case_id": "..."}

Usage:
  python prepare_ft_data.py \
      --json /scratch/.../data/train/medreason_train_selection.json \
      --imgs /scratch/.../data/train/imgs \
      --out  /scratch/.../ft_data \
      --val-n 1000
"""
import argparse, json, os, random

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


def build_example(case, imgs_dir):
    img = os.path.join(imgs_dir, os.path.basename(case["image_path"]))
    if not os.path.exists(img):
        return None
    qtype = case.get("question type")
    if qtype == "mcq":
        gold = str(case.get("answer", "")).strip().upper()
        if gold not in {"A", "B", "C", "D", "E"}:
            return None
        return {"image": img, "prompt": mcq_prompt(case), "target": gold,
                "task_type": "mcq", "case_id": case["case_id"]}
    else:
        answer = str(case.get("answer", "")).strip()
        trace = str(case.get("visual_description", "")).strip() or answer
        if not answer:
            return None
        target = json.dumps({"reasoning_trace": trace, "answer": answer}, ensure_ascii=False)
        return {"image": img, "prompt": open_prompt(case), "target": target,
                "task_type": "open", "case_id": case["case_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = json.load(open(args.json))["cases"]

    examples = [e for e in (build_example(c, args.imgs) for c in cases) if e]
    random.seed(args.seed)
    random.shuffle(examples)

    val = examples[:args.val_n]
    train = examples[args.val_n:]

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(train, os.path.join(args.out, "train.jsonl"))
    dump(val, os.path.join(args.out, "val.jsonl"))

    from collections import Counter
    print(f"total usable: {len(examples)}")
    print(f"train: {len(train)}  ({Counter(e['task_type'] for e in train)})")
    print(f"val:   {len(val)}  ({Counter(e['task_type'] for e in val)})")
    print(f"written to {args.out}/train.jsonl and val.jsonl")


if __name__ == "__main__":
    main()
