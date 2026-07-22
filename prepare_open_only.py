#!/usr/bin/env python3
"""
prepare_open_only.py — build OPEN-ENDED-ONLY LoRA training data.

Rationale: our v4 adapter was trained on a mostly-MCQ mixture, which taught the
model to be terse and degraded open-ended quality (traces 44 -> 20 words, GT
1.588 -> 1.454). Training an adapter on ONLY open-ended cases removes that
pressure and teaches the reference answer style directly, which is what the
GT judge compares against.

Target format matches inference exactly:
  {"reasoning_trace": "<modality of the <organ> system. <visual description>>",
   "answer": "<gold answer>"}

Uses the SAME held-out case_ids as ft_data_v3/val.jsonl so validation stays clean
(no leakage between this training set and the val set we measure on).

Usage:
  python prepare_open_only.py \
    --json data/train/medreason_train_selection.json \
    --imgs data/train/imgs \
    --val-ref ft_data_v3/val.jsonl \
    --out ft_data_open
"""
import argparse, json, os, random
from collections import Counter

OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. Examine the image carefully and answer with clinical precision.")


def open_prompt(question):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {question}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image '
            "(modality, anatomical location, shape, margins, density or signal intensity). "
            "Do NOT state anything you cannot see.\n"
            '- "answer": a single specific, committed clinical statement that directly answers '
            "the question.")


def build(case, imgs_dir):
    if case.get("question type") != "open-ended":
        return None
    img = os.path.join(imgs_dir, os.path.basename(case["image_path"]))
    if not os.path.exists(img):
        return None
    answer = str(case.get("answer", "")).strip()
    if not answer:
        return None
    vd = str(case.get("visual_description", "")).strip()
    modality = str(case.get("primary_modality", case.get("modality", ""))).strip()
    organ = str(case.get("organ system", "")).strip()
    if modality and organ:
        lead = f"{modality} of the {organ.lower()} system. "
    elif modality:
        lead = f"{modality} image. "
    elif organ:
        lead = f"Imaging of the {organ.lower()} system. "
    else:
        lead = ""
    trace = (lead + vd).strip() if vd else (lead + "The relevant structures are visible.").strip()
    target = json.dumps({"reasoning_trace": trace, "answer": answer}, ensure_ascii=False)
    return {"image": img, "prompt": open_prompt(case["question"]), "target": target,
            "task_type": "open", "case_id": case["case_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--val-ref", required=True,
                    help="existing val.jsonl whose case_ids must be EXCLUDED from training")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # case_ids already used for validation -> never train on them
    val_ids = set()
    for line in open(args.val_ref):
        r = json.loads(line)
        if r.get("case_id"):
            val_ids.add(r["case_id"])
    print(f"Excluding {len(val_ids)} validation case_ids from training", flush=True)

    cases = json.load(open(args.json))["cases"]
    ex = [e for e in (build(c, args.imgs) for c in cases) if e]
    train = [e for e in ex if e["case_id"] not in val_ids]
    held = [e for e in ex if e["case_id"] in val_ids]

    random.seed(0)
    random.shuffle(train)

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(train, os.path.join(args.out, "train.jsonl"))
    dump(held, os.path.join(args.out, "val.jsonl"))
    print(f"open-ended total: {len(ex)}")
    print(f"train (open only, val excluded): {len(train)}")
    print(f"val  (same cases as ft_data_v3 val): {len(held)}")
    if train:
        print("\nSAMPLE target:\n", train[0]["target"][:280])


if __name__ == "__main__":
    main()
