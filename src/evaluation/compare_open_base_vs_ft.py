#!/usr/bin/env python3
"""
compare_open_base_vs_ft.py — settle the hybrid question with data.

The organizer leaderboard showed the OFF-THE-SHELF Qwen2.5-VL baseline beating our
fine-tuned model on BOTH open-ended metrics (GT 1.588 vs 1.454, VA 2.696 vs 2.296),
while our fine-tune massively wins MCQ (96.22% vs 29.43%).

This script scores open-ended answers from BOTH models on the same val cases with
the same proxies, so we confirm the effect locally before rebuilding the Docker.

Usage:
  python compare_open_base_vs_ft.py \
    --val ft_data_v3/val.jsonl \
    --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
    --ft   models/Qwen2.5-VL-merged \
    --n 202
"""
import argparse, json, os, re, string
from collections import Counter
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

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


def score_model(tag, model_path, rows, root, sbert):
    print(f"\n===== {tag}: {model_path} =====", flush=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = VLModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True).eval()
    f1s, cos, trace_len = [], [], []
    for i, r in enumerate(rows):
        img = Image.open(resolve(r["image"], root)).convert("RGB")
        q = r["prompt"].split("Question:")[-1].split("Respond ONLY")[0].strip()
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": open_prompt(q)}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False)
        dec = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ans, trace = extract(dec)
        gold = json.loads(r["target"])["answer"] if r["target"].startswith("{") else r["target"]
        f1s.append(token_f1(ans, gold))
        trace_len.append(len(trace.split()))
        if sbert is not None:
            from sentence_transformers import util
            e = sbert.encode([ans, gold], convert_to_tensor=True)
            cos.append(float(util.cos_sim(e[0], e[1])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}  F1={np.mean(f1s):.3f}", flush=True)
    print(f"\n--- {tag} RESULTS on {len(rows)} open cases ---")
    print(f"  mean token F1:      {np.mean(f1s):.3f}")
    if cos:
        print(f"  mean embed cos sim: {np.mean(cos):.3f}")
    print(f"  mean trace length:  {np.mean(trace_len):.1f} words")
    del model
    torch.cuda.empty_cache()
    return np.mean(f1s), (np.mean(cos) if cos else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--n", type=int, default=202)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val)
            if json.loads(l)["task_type"] == "open"][:args.n]
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sbert = None

    b_f1, b_cos = score_model("BASE Qwen2.5-VL (off-the-shelf)", args.base, rows, args.imgs_root, sbert)
    f_f1, f_cos = score_model("FINE-TUNED (merged v4)", args.ft, rows, args.imgs_root, sbert)

    print("\n" + "=" * 60)
    print("HYBRID DECISION")
    print("=" * 60)
    print(f"  BASE       : F1 {b_f1:.3f}  cos {b_cos:.3f}")
    print(f"  FINE-TUNED : F1 {f_f1:.3f}  cos {f_cos:.3f}")
    if b_f1 > f_f1 or b_cos > f_cos:
        print("\n  -> BASE is better on open-ended. USE HYBRID:")
        print("     MCQ  -> fine-tuned model")
        print("     OPEN -> base model")
    else:
        print("\n  -> Fine-tuned holds up locally; organizer gap may come from their prompts.")


if __name__ == "__main__":
    main()
