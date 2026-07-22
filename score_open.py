#!/usr/bin/env python3
"""
score_open.py — approximate open-ended GT correctness on the FULL open-ended
val set (not just 15 cases), so we can compare models with a real number.

The official metric uses Llama-3.1-70B as judge (we can't run that here), so we
approximate with two automatic proxies:
  1. token F1  (overlap of content words, like SQuAD)
  2. embedding cosine similarity (semantic), if sentence-transformers is available

These are PROXIES, not the official score, but on 200 cases they give a far more
stable comparison than eyeballing 15. Report mean over all open cases.

Usage:
  python score_open.py --val ft_data_v3/val.jsonl \
     --model <base> --adapter lora_medreason_v4 --imgs-root . --n 202
"""
import argparse, json, os, re, string
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


def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return s.split()


def token_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return str(json.loads(m.group(0)).get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    return raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--n", type=int, default=202)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val) if json.loads(l)["task_type"] == "open"][:args.n]
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading {args.adapter}…", flush=True)
    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # optional embedding model
    embed = None
    try:
        from sentence_transformers import SentenceTransformer, util
        embed = SentenceTransformer("all-MiniLM-L6-v2")
        print("Using embedding similarity + token F1", flush=True)
    except Exception:
        print("sentence-transformers not available; using token F1 only", flush=True)

    f1s, sims = [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
            msg = [{"role": "user", "content": [{"type": "image", "image": img},
                    {"type": "text", "text": r["prompt"]}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = proc(text=[t], images=[img], return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=200, do_sample=False)
            dec = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(dec)
            gold = json.loads(r["target"])["answer"] if r["target"].startswith("{") else r["target"]
            f1s.append(token_f1(pred, gold))
            if embed is not None:
                e = embed.encode([pred, gold], convert_to_tensor=True)
                sims.append(float(util.cos_sim(e[0], e[1])))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rows)}  running F1={sum(f1s)/len(f1s):.3f}", flush=True)

    print(f"\n=== {args.adapter} on {len(rows)} open cases ===")
    print(f"mean token F1:      {sum(f1s)/len(f1s):.3f}")
    if sims:
        print(f"mean embed cos sim: {sum(sims)/len(sims):.3f}")


if __name__ == "__main__":
    main()
