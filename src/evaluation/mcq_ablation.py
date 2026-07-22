#!/usr/bin/env python3
"""
mcq_ablation.py — figure out WHY MCQ accuracy is low.
Compares three MCQ prompting styles on the same train cases:
  A) json   : ask for JSON {reasoning_trace, answer}  (current custom_system)
  B) letter : ask for ONLY the option letter           (simple, robust)
  C) letter_then_reason : letter first line, reason after
Reports accuracy for each so we can pick the best.
"""
import argparse, json, os, re, random, time
import torch
from transformers import AutoProcessor
from PIL import Image
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

SYS = ("You are an expert radiologist answering a multiple-choice question about "
       "a medical image. Examine the image carefully, then choose the single best option.")


def opts_block(case):
    return "\n".join(f"{k}. {case[k]}" for k in ("A", "B", "C", "D", "E") if k in case)


def prompt_json(case):
    labels = "/".join(k for k in ("A", "B", "C", "D", "E") if k in case)
    return (f"{SYS}\n\nQuestion: {case['question']}\n\nOptions:\n{opts_block(case)}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer". '
            f'The "answer" must be exactly one label, one of {labels}.')


def prompt_letter(case):
    return (f"{SYS}\n\nQuestion: {case['question']}\n\nOptions:\n{opts_block(case)}\n\n"
            "Answer with ONLY the single letter of the best option (A, B, C, D, or E). "
            "Output just the letter, nothing else.")


def prompt_letter_then_reason(case):
    return (f"{SYS}\n\nQuestion: {case['question']}\n\nOptions:\n{opts_block(case)}\n\n"
            "On the first line output ONLY the letter of the best option (A-E). "
            "Then on the next line briefly justify it from the image.")


def extract_letter(text, case, style):
    labels = [k for k in ("A", "B", "C", "D", "E") if k in case]
    t = text.strip()
    if style == "json":
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                a = str(json.loads(m.group(0)).get("answer", "")).strip().upper()
                if a and a[0] in labels:
                    return a[0]
            except json.JSONDecodeError:
                pass
    # letter styles (and json fallback): take the FIRST standalone A-E
    m = re.match(r"\s*\(?([A-E])\b", t.upper())
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", t.upper())
    return m.group(1) if m else labels[0]


def gen(model, proc, case, imgs, prompt, max_new):
    img = Image.open(os.path.join(imgs, os.path.basename(case["image_path"]))).convert("RGB")
    msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    cases = [c for c in json.load(open(args.json))["cases"] if c.get("question type") == "mcq"]
    random.seed(0); random.shuffle(cases); cases = cases[:args.n]

    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(args.model)
    model.eval()

    styles = [
        ("json", prompt_json, 512),
        ("letter", prompt_letter, 8),
        ("letter_then_reason", prompt_letter_then_reason, 160),
    ]
    for name, fn, max_new in styles:
        correct, t0 = 0, time.time()
        for i, c in enumerate(cases):
            raw = gen(model, proc, c, args.imgs, fn(c), max_new)
            if extract_letter(raw, c, name) == str(c.get("answer", "")).strip().upper():
                correct += 1
        acc = correct / len(cases)
        print(f"[{name:20s}] acc={acc:.3f} on {len(cases)} cases ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
