#!/usr/bin/env python3
"""
prepare_ft_data_v3.py — polished fine-tuning data for MedReason.

Improvements over v2:
  * Removes the parroting phrase "These findings support the conclusion."
  * Open-ended reasoning trace is ANATOMY-ANCHORED: it leads with the modality
    and visible finding (drawn from visual_description) then commits to the gold
    answer. This serves RVF (image-grounded faithfulness) without a separate,
    hallucination-prone verification turn.
  * Keeps gold `answer` as the target answer (serves GT).
  * Upweights open-ended (default 3x) to balance the 81/19 split.
  * Naturally retains negative/absent findings that already exist in gold answers
    (we do NOT inject artificial abstentions, which would hurt GT).
"""
import argparse, json, os, random
from collections import Counter

MCQ_SYSTEM = ("You are an expert radiologist answering a multiple-choice question about a "
              "medical image. Examine the image carefully, then choose the single best option.")
OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. First identify the modality and region you can see, then give "
               "a precise, image-grounded answer.")


def mcq_prompt(case):
    opts = "\n".join(f"{k}. {case[k]}" for k in ("A", "B", "C", "D", "E") if k in case)
    return (f"{MCQ_SYSTEM}\n\nQuestion: {case['question']}\n\nOptions:\n{opts}\n\n"
            "Answer with ONLY the single letter of the best option (A, B, C, D, or E). "
            "Output just the letter, nothing else.")


def open_prompt(case):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {case['question']}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences. State the imaging modality and anatomical '
            'region first, then the specific visible finding. Describe ONLY what is visible; '
            'do not assert findings you cannot see.\n'
            '- "answer": one specific, committed clinical statement that answers the question.')


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
        modality = str(case.get("primary_modality", case.get("modality", ""))).strip()
        organ = str(case.get("organ system", "")).strip()
        # anatomy-anchored trace, NO parrot phrase.
        # Build a clean lead like "MRI of the cardiovascular system." then a space.
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
    print(f"train: {len(weighted)} ({Counter(e['task_type'] for e in weighted)})")
    print(f"val:   {len(val)} ({Counter(e['task_type'] for e in val)})")
    # show a sample open target so we can eyeball the new trace style
    for r in weighted:
        if r["task_type"] == "open":
            print("\nSAMPLE open target:\n", r["target"][:300])
            break


if __name__ == "__main__":
    main()
