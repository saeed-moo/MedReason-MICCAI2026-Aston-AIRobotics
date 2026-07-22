#!/usr/bin/env python3
"""
validate_hybrid_v2.py — hybrid toggle validation with EXACT base-model prompting
(fixes the question-extraction that caused a small F1 discrepancy in v1).

Reconstructs the open-ended prompt identically to compare_open_base_vs_ft.py so the
open-ended F1 is directly comparable to the base model's standalone 0.171.
"""
import argparse, json, os, re, string
from collections import Counter
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

MCQ_SYSTEM = ("You are an expert radiologist answering a multiple-choice question about a "
              "medical image. Examine the image carefully, then choose the single best option.")
OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. Examine the image carefully and answer with clinical precision.")


def get_question(prompt):
    """Robustly pull the question text out of a stored prompt."""
    if "Question:" in prompt:
        after = prompt.split("Question:", 1)[1]
        # cut at the first instruction marker that follows the question
        for marker in ("\n\nOptions:", "\n\nRespond ONLY", "\n\nAnswer with", "Respond ONLY", "Answer with"):
            if marker in after:
                after = after.split(marker)[0]
        return after.strip()
    return prompt.strip()


def mcq_prompt(q):
    return (f"{MCQ_SYSTEM}\n\nQuestion: {q}\n\n"
            "Answer with ONLY the single letter of the best option (A, B, C, D, or E). "
            "Output just the letter, nothing else.")


def open_prompt(q):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {q}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image '
            "(modality, anatomical location, shape, margins, density or signal intensity). "
            "Do NOT state anything you cannot see.\n"
            '- "answer": a single specific, committed clinical statement that directly answers '
            "the question.")


def norm(s):
    s = s.lower()
    return "".join(c for c in s if c not in string.punctuation).split()


def token_f1(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def extract(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            return str(o.get("answer", "")).strip(), str(o.get("reasoning_trace", "")).strip()
        except json.JSONDecodeError:
            pass
    return raw.strip(), raw.strip()


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def gen(model, proc, img, prompt, max_new):
    msg = [{"role": "user", "content": [{"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--mcq-n", type=int, default=400)
    ap.add_argument("--open-n", type=int, default=202)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val)]
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    print("Loading base + adapter…", flush=True)
    model = VLModel.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="ft")
    model.eval()

    # MCQ, adapter ON
    model.set_adapter("ft")
    mcq = [r for r in rows if r["task_type"] == "mcq"][:args.mcq_n]
    correct, preds = 0, []
    for r in mcq:
        img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
        dec = gen(model, proc, img, mcq_prompt(get_question(r["prompt"])), 8)
        m = re.search(r"\b([A-E])\b", dec.upper())
        p = m.group(1) if m else "?"
        preds.append(p)
        if p == r["target"].strip().upper():
            correct += 1
    mcq_acc = correct / len(mcq)

    # OPEN, adapter OFF
    with model.disable_adapter():
        opn = [r for r in rows if r["task_type"] == "open"][:args.open_n]
        f1s, tlens = [], []
        for r in opn:
            img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
            dec = gen(model, proc, img, open_prompt(get_question(r["prompt"])), 256)
            ans, trace = extract(dec)
            gold = json.loads(r["target"])["answer"] if r["target"].startswith("{") else r["target"]
            f1s.append(token_f1(ans, gold))
            tlens.append(len(trace.split()))

    print("\n" + "=" * 55)
    print("HYBRID TOGGLE VALIDATION (v2, exact prompts)")
    print("=" * 55)
    print(f"  MCQ  (adapter ON):  acc = {mcq_acc:.3f}  ({len(mcq)} cases)")
    print(f"       letters: {dict(sorted(Counter(preds).items()))}")
    print(f"  OPEN (adapter OFF): F1  = {np.mean(f1s):.3f}  ({len(opn)} cases)")
    print(f"       mean trace length: {np.mean(tlens):.1f} words")
    print("\n  Base standalone was: MCQ n/a | OPEN F1 0.171, 44.3 words")
    print("  Fine-tuned standalone was: MCQ 0.945 | OPEN F1 0.166, 19.9 words")


if __name__ == "__main__":
    main()
