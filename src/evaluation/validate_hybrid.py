#!/usr/bin/env python3
"""
validate_hybrid.py — prove the adapter-TOGGLE hybrid works before building Docker.

One base Qwen2.5-VL + the v4 LoRA adapter. Per task:
  MCQ  -> adapter ENABLED  (fine-tuned; strong closed-ended)
  OPEN -> adapter DISABLED (pure base; richer traces, better GT/VA)

Confirms on val that:
  - MCQ accuracy stays high (~0.94) with adapter on
  - Open-ended matches the BASE model (F1 ~0.171, longer traces) with adapter off

This is the exact behaviour the hybrid Docker will implement.

Usage (Isambard GPU):
  python validate_hybrid.py \
    --val ft_data_v3/val.jsonl \
    --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
    --adapter lora_medreason_v4 \
    --mcq-n 400 --open-n 202
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


def mcq_prompt(question):
    return (f"{MCQ_SYSTEM}\n\nQuestion: {question}\n\n"
            "Answer with ONLY the single letter of the best option (A, B, C, D, or E). "
            "Output just the letter, nothing else.")


def open_prompt(question):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {question}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image '
            "(modality, anatomical location, shape, margins, density or signal intensity). "
            "Do NOT state anything you cannot see.\n"
            '- "answer": a single specific, committed clinical statement.')


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
    print("Loading base + adapter (named 'ft')…", flush=True)
    model = VLModel.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="ft")
    model.eval()

    # ---- MCQ with adapter ENABLED ----
    model.set_adapter("ft")                 # adapter ON
    mcq = [r for r in rows if r["task_type"] == "mcq"][:args.mcq_n]
    correct, preds = 0, []
    for r in mcq:
        img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
        q = r["prompt"].split("Question:")[-1].split("Answer with")[0].strip()
        dec = gen(model, proc, img, mcq_prompt(q), 8)
        m = re.search(r"\b([A-E])\b", dec.upper())
        p = m.group(1) if m else "?"
        preds.append(p)
        if p == r["target"].strip().upper():
            correct += 1
    mcq_acc = correct / len(mcq) if mcq else 0.0

    # ---- OPEN with adapter DISABLED (pure base) ----
    with model.disable_adapter():           # adapter OFF -> base behaviour
        opn = [r for r in rows if r["task_type"] == "open"][:args.open_n]
        f1s, tlens = [], []
        for r in opn:
            img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
            q = r["prompt"].split("Question:")[-1].split("Respond ONLY")[0].strip()
            dec = gen(model, proc, img, open_prompt(q), 256)
            ans, trace = extract(dec)
            gold = json.loads(r["target"])["answer"] if r["target"].startswith("{") else r["target"]
            f1s.append(token_f1(ans, gold))
            tlens.append(len(trace.split()))

    print("\n" + "=" * 55)
    print("HYBRID TOGGLE VALIDATION")
    print("=" * 55)
    print(f"  MCQ  (adapter ON):  acc = {mcq_acc:.3f}  on {len(mcq)} cases")
    print(f"       letters: {dict(sorted(Counter(preds).items()))}")
    print(f"  OPEN (adapter OFF): F1  = {np.mean(f1s):.3f}  on {len(opn)} cases")
    print(f"       mean trace length: {np.mean(tlens):.1f} words")
    print("\n  Targets: MCQ ~0.945 (fine-tuned) | OPEN ~0.171 F1, ~44 words (base)")
    print("  If both hold, the toggle works -> build hybrid Docker.")


if __name__ == "__main__":
    main()
