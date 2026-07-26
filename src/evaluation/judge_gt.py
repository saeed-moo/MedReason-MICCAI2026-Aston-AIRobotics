#!/usr/bin/env python3
"""
judge_gt.py — LOCAL LLM judge that approximates the challenge's open-ended GT metric.

WHY THIS EXISTS
Token-F1 and MiniLM cosine cannot resolve differences in open-ended medical
answers: four very different systems all scored within 0.016 of each other,
while the official GT judge spreads teams from 0.51 to 2.07. We were optimising
blind. This scores answers 0-4 with an LLM using a rubric mirroring the official
GT definition (agreement with the reference answer).

CALIBRATION ANCHOR
The organizers published that the off-the-shelf Qwen2.5-VL baseline scores
GT 1.588. Judge the BASE model's answers first: if this judge returns roughly
1.5-1.7, it is calibrated and can be trusted to compare other configs. If it
returns something far off (e.g. 3.2 or 0.4), the rubric is mis-scaled --
adjust before drawing conclusions from it.

Usage (Isambard GPU):
  # 1. calibrate on base answers
  python judge_gt.py --answers answers_base.jsonl --out judged_base.jsonl
  # 2. then judge any other config and compare
  python judge_gt.py --answers answers_loraopen.jsonl --out judged_loraopen.jsonl

Judge model: any local instruct LLM. Default Qwen2.5-32B-Instruct (Apache-2.0,
not gated, fits bf16 on a 96GB GH200). Use --judge Qwen/Qwen2.5-14B-Instruct
for a faster/smaller run.
"""
import argparse, json, re
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RUBRIC = """You are grading a medical visual question answering system.

You are given a QUESTION about a medical image, the REFERENCE ANSWER written by an expert, and the MODEL ANSWER produced by the system.

Score how well the MODEL ANSWER agrees with the REFERENCE ANSWER, on a scale of 0 to 4:

4 = Fully correct. Conveys the same clinical meaning as the reference, with the same specificity.
3 = Mostly correct. Same core finding, but slightly less specific or missing a minor detail.
2 = Partially correct. Related to the reference and not wrong, but too vague, incomplete, or only captures part of it.
1 = Minimally related. Touches the right general area (e.g. correct organ or modality) but does not give the reference finding.
0 = Incorrect, contradictory, or irrelevant.

Judge only clinical agreement with the reference. Do not reward length, fluency, or hedging. A short answer that matches the reference scores high. A long answer that misses the reference finding scores low.

QUESTION: {question}

REFERENCE ANSWER: {gold}

MODEL ANSWER: {pred}

Respond with ONLY a compact JSON object: {{"score": <0-4 integer>, "reason": "<one short sentence>"}}"""


def parse_score(text):
    m = re.search(r'"score"\s*:\s*([0-4])', text)
    if m:
        return int(m.group(1)), text
    m = re.search(r'\b([0-4])\b', text)
    if m:
        return int(m.group(1)), text
    return None, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="jsonl from gen_open_answers.py")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.answers)]
    if args.n:
        rows = rows[:args.n]
    print(f"Judging {len(rows)} answers with {args.judge}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("Judge loaded.", flush=True)

    scores, unparsed = [], 0
    out = open(args.out, "w", encoding="utf-8")
    for i, r in enumerate(rows):
        prompt = RUBRIC.format(question=r["question"], gold=r["gold"],
                               pred=r["pred_answer"])
        msg = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = tok([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=80, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        dec = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        s, raw = parse_score(dec)
        if s is None:
            unparsed += 1
            s = 0
        scores.append(s)
        rec = dict(r)
        rec["gt_score"] = s
        rec["judge_raw"] = raw.strip()[:300]
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  running mean GT={np.mean(scores):.3f}", flush=True)
    out.close()

    hist = Counter(scores)
    print("\n" + "=" * 56)
    print(f"LOCAL GT JUDGE — {args.answers}")
    print("=" * 56)
    print(f"  mean GT score (0-4): {np.mean(scores):.3f}   over {len(scores)} cases")
    print(f"  distribution: " + "  ".join(f"{k}:{hist.get(k,0)}" for k in range(5)))
    print(f"  unparsed judge outputs (scored 0): {unparsed}")
    print(f"\n  Saved per-case scores to {args.out}")
    print("\n  CALIBRATION: organizers report the off-the-shelf Qwen2.5-VL baseline")
    print("  at GT 1.588. If this run is on BASE answers and lands near 1.5-1.7,")
    print("  the judge is calibrated and can be trusted for comparisons.")


if __name__ == "__main__":
    main()
