#!/usr/bin/env python3
"""
test_gt_prompts.py — push open-ended GT by matching the REFERENCE ANSWER STYLE.

The GT judge (Llama-3.1-70B) compares our answer to the gold reference. Gold
answers read like a radiology report impression line: direct, specific, no
hedging, often a noun phrase ("The right coronary cusp", "Tortuous aortic arch",
"Vascular etiology"). Our base model answers in fuller sentences.

This tests several answer-style prompts on the BASE model (adapter off = the
config our hybrid uses for open-ended) and scores each with the same proxies,
so we can see which style best matches the references.

Usage (Isambard GPU):
  python test_gt_prompts.py --val ft_data_v3/val.jsonl \
    --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
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

SYS = ("You are an expert radiologist answering an open-ended question about a "
       "medical image. Examine the image carefully and answer with clinical precision.")


def get_q(prompt):
    if "Question:" in prompt:
        a = prompt.split("Question:", 1)[1]
        for m in ("\n\nOptions:", "\n\nRespond ONLY", "\n\nAnswer with", "Respond ONLY", "Answer with"):
            if m in a:
                a = a.split(m)[0]
        return a.strip()
    return prompt.strip()


# ---- answer-style variants -------------------------------------------------
def p_current(q):
    """A: current hybrid prompt (baseline for comparison)."""
    return (f"{SYS}\n\nQuestion: {q}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image '
            "(modality, anatomical location, shape, margins, density or signal intensity). "
            "Do NOT state anything you cannot see.\n"
            '- "answer": a single specific, committed clinical statement that directly answers '
            "the question.")


def p_impression(q):
    """B: match the reference style -- radiology impression line."""
    return (f"{SYS}\n\nQuestion: {q}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing the modality, anatomical region and '
            "the specific visible finding. Describe only what is visible.\n"
            '- "answer": write it exactly as the impression line of a radiology report \u2014 the '
            "specific finding or entity itself, stated directly. No preamble, no hedging, no "
            "\"the image shows\". Just the finding (e.g. \"Tortuous aortic arch\", "
            "\"Diffusely low T2 signal intensity\").")


def p_named_entity(q):
    """C: force naming the specific structure/finding/diagnosis."""
    return (f"{SYS}\n\nQuestion: {q}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences on modality, region and the visible finding.\n'
            '- "answer": name the specific anatomical structure, finding, pattern or diagnosis '
            "that answers the question. Be as precise as the evidence allows \u2014 give the exact "
            "term a radiologist would use, not a general description.")


def p_detailed(q):
    """D: richer answer -- more clinical specifics (partial-credit strategy)."""
    return (f"{SYS}\n\nQuestion: {q}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences on modality, region and the visible finding.\n'
            '- "answer": state the specific finding together with its key defining '
            "characteristics (location, distribution, signal/density, margins) in one dense "
            "clinical sentence. Include the precise terminology.")


VARIANTS = [("A_current", p_current), ("B_impression", p_impression),
            ("C_named_entity", p_named_entity), ("D_detailed", p_detailed)]


def norm(s):
    return "".join(ch for ch in s.lower() if ch not in string.punctuation).split()


def token_f1(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return 0.0
    c = Counter(p) & Counter(g)
    s = sum(c.values())
    if s == 0:
        return 0.0
    pr, rc = s / len(p), s / len(g)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--n", type=int, default=202)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val)
            if json.loads(l)["task_type"] == "open"][:args.n]
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    print("Loading base model (open-ended config, no adapter)…", flush=True)
    model = VLModel.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                    trust_remote_code=True).to("cuda").eval()

    try:
        from sentence_transformers import SentenceTransformer, util
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sbert, util = None, None

    results = {}
    for name, fn in VARIANTS:
        f1s, cos, alen = [], [], []
        for i, r in enumerate(rows):
            img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
            msg = [{"role": "user", "content": [{"type": "image", "image": img},
                    {"type": "text", "text": fn(get_q(r["prompt"]))}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            e = proc(text=[t], images=[img], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                o = model.generate(**e, max_new_tokens=256, do_sample=False)
            dec = proc.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)
            ans, _ = extract(dec)
            gold = json.loads(r["target"])["answer"] if r["target"].startswith("{") else r["target"]
            f1s.append(token_f1(ans, gold))
            alen.append(len(ans.split()))
            if sbert is not None:
                emb = sbert.encode([ans, gold], convert_to_tensor=True)
                cos.append(float(util.cos_sim(emb[0], emb[1])))
            if (i + 1) % 100 == 0:
                print(f"  [{name}] {i+1}/{len(rows)} F1={np.mean(f1s):.3f}", flush=True)
        results[name] = (np.mean(f1s), np.mean(cos) if cos else 0.0, np.mean(alen))
        print(f"  [{name}] done: F1={results[name][0]:.3f} cos={results[name][1]:.3f} "
              f"answer_len={results[name][2]:.1f}w", flush=True)

    print("\n" + "=" * 62)
    print("OPEN-ENDED ANSWER-STYLE COMPARISON (base model, 202 val cases)")
    print("=" * 62)
    print(f"{'variant':<16}{'tokenF1':>10}{'cos_sim':>10}{'ans_words':>12}")
    for name, (f, c, l) in results.items():
        print(f"{name:<16}{f:>10.3f}{c:>10.3f}{l:>12.1f}")
    best = max(results.items(), key=lambda kv: kv[1][1])
    print(f"\nBest by embedding similarity: {best[0]}")
    print("NOTE: these are PROXIES for the organizer GT judge, not official 0-4 scores.")


if __name__ == "__main__":
    main()
