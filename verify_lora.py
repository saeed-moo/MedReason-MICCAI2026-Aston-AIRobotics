#!/usr/bin/env python3
"""
verify_lora.py — sanity-check the fine-tuned MedReason model on held-out val.
Loads base model + LoRA adapter, then reports:
  1. MCQ accuracy + predicted-letter distribution (catches answer-letter bias)
  2. Gold-letter distribution (to compare against predictions)
  3. Open-ended examples (gold vs model) to confirm reasoning survived

Usage:
  python verify_lora.py --val ft_data/val.jsonl \
    --model <base path> --adapter lora_medreason --mcq-n 400 --open-n 15
"""
import argparse, json, os, re
from collections import Counter
import torch
from transformers import AutoProcessor
from peft import PeftModel
from PIL import Image
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def gen(model, proc, row, root, max_new):
    image = Image.open(resolve(row["image"], root)).convert("RGB")
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": row["prompt"]}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--mcq-n", type=int, default=400)
    ap.add_argument("--open-n", type=int, default=15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val)]
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print("Loading base + adapter…", flush=True)
    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    mcq = [r for r in rows if r["task_type"] == "mcq"][:args.mcq_n]
    preds, golds, correct = [], [], 0
    with torch.no_grad():
        for r in mcq:
            dec = gen(model, proc, r, args.imgs_root, 8)
            m = re.search(r"\b([A-E])\b", dec.upper())
            pred = m.group(1) if m else "?"
            gold = r["target"].strip().upper()
            preds.append(pred); golds.append(gold)
            if pred == gold:
                correct += 1
    acc = correct / len(mcq) if mcq else 0
    print(f"\n=== MCQ accuracy: {acc:.3f} on {len(mcq)} val cases ===")
    print("predicted letters:", dict(sorted(Counter(preds).items())))
    print("gold letters     :", dict(sorted(Counter(golds).items())))
    print("(if predicted is heavily skewed vs gold -> shortcut/bias warning)\n")

    opn = [r for r in rows if r["task_type"] == "open"][:args.open_n]
    print(f"=== {len(opn)} open-ended examples (gold vs model) ===")
    with torch.no_grad():
        for r in opn:
            dec = gen(model, proc, r, args.imgs_root, 256)
            gold = json.loads(r["target"])
            print("-" * 70)
            print("GOLD ans :", gold.get("answer", "")[:140])
            print("MODEL    :", dec[:240])


if __name__ == "__main__":
    main()
