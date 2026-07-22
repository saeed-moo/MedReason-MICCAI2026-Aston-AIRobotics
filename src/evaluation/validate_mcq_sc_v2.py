#!/usr/bin/env python3
"""
validate_mcq_sc_v2.py — MCQ self-consistency test, FIXED.
Uses the stored val prompt DIRECTLY (it already contains system + question +
options + instruction), avoiding the options-stripping bug of v1.

Greedy vs vote@k on MCQ val cases, using base + adapter (fine-tuned MCQ path).

Usage (Isambard GPU):
  python validate_mcq_sc_v2.py \
    --val ft_data_v3/val.jsonl \
    --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
    --adapter lora_medreason_v4 --n 400 --k 5 --temp 0.7
"""
import argparse, json, os, re
from collections import Counter
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def letter_of(text):
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val) if json.loads(l)["task_type"] == "mcq"][:args.n]
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    print("Loading base + adapter…", flush=True)
    model = VLModel.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    greedy_ok, vote_ok, changed = 0, 0, 0
    for i, r in enumerate(rows):
        img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
        # USE THE STORED PROMPT DIRECTLY (already has options + instruction)
        prompt = r["prompt"]
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        gold = r["target"].strip().upper()

        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=8, do_sample=False)
        greedy = letter_of(proc.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))

        votes = []
        with torch.inference_mode():
            for _ in range(args.k):
                s = model.generate(**enc, max_new_tokens=8, do_sample=True,
                                   temperature=args.temp, top_p=0.9)
                votes.append(letter_of(proc.decode(s[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
        vc = Counter([v for v in votes if v != "?"]).most_common(1)
        vote = vc[0][0] if vc else greedy

        if greedy == gold:
            greedy_ok += 1
        if vote == gold:
            vote_ok += 1
        if vote != greedy:
            changed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}  greedy={greedy_ok/(i+1):.3f}  vote={vote_ok/(i+1):.3f}", flush=True)

    n = len(rows)
    print("\n" + "=" * 50)
    print("MCQ SELF-CONSISTENCY (v2, correct prompts)")
    print("=" * 50)
    print(f"  greedy accuracy:     {greedy_ok/n:.4f}")
    print(f"  vote@{args.k} accuracy:     {vote_ok/n:.4f}  (temp {args.temp})")
    print(f"  delta:               {(vote_ok-greedy_ok)/n:+.4f}   ({changed} changed)")
    print(f"  voting cost:         ~{args.k+1}x MCQ inference time")
    print("\n  Adopt voting only if delta is clearly positive.")


if __name__ == "__main__":
    main()
