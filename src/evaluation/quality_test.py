#!/usr/bin/env python3
"""
quality_test.py — measure the improved MedReason prompts on the TRAIN split
(which has answers) so we know if they beat the baseline BEFORE submitting.

Runs the same MCQ/open prompts as custom_system.py, scores MCQ by exact-match,
and prints open-ended examples so you can eyeball specificity + grounding.

Usage (on a GH200 via Slurm or srun):
  python quality_test.py \
      --json  /scratch/.../data/train/medreason_train_selection.json \
      --imgs  /scratch/.../data/train/imgs \
      --model /scratch/.../models/Qwen2.5-VL \
      --mcq-n 300 --open-n 40
"""
import argparse, json, os, re, sys, time, random
import torch
from transformers import AutoProcessor
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


# --- prompts: copy of the competitive prompts in custom_system.py ----------
MCQ_SYSTEM = ("You are an expert radiologist answering a multiple-choice question "
              "about a medical image. Examine the image carefully, then choose the single best option.")
OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. Examine the image carefully and answer with clinical precision.")


def mcq_prompt(case):
    opts = "\n".join(f"{k}. {case[k]}" for k in ("A", "B", "C", "D", "E") if k in case)
    labels = "/".join(k for k in ("A", "B", "C", "D", "E") if k in case)
    return (f"{MCQ_SYSTEM}\n\nQuestion: {case['question']}\n\nOptions:\n{opts}\n\n"
            "Decide which option is best supported by the visible image evidence.\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": one or two sentences citing the specific visual finding '
            "(location, shape, margin, density/intensity) that decides the answer. Mention only what is visible.\n"
            f'- "answer": exactly one option label, one of {labels}. No other text.')


def open_prompt(case):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {case['question']}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image '
            "(modality, anatomical location, shape, margins, density or signal intensity). "
            "Do NOT state anything you cannot see. Do NOT add generic filler.\n"
            '- "answer": a single specific, committed clinical statement that directly answers '
            "the question. Be concrete. Do NOT hedge with vague phrases like 'possible abnormality' "
            "unless the image genuinely shows no diagnostic evidence.")


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse(text):
    m = _JSON_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            return str(obj.get("reasoning_trace", "")).strip(), str(obj.get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    return text.strip(), text.strip()


def norm_mcq(answer, case):
    labels = [k for k in ("A", "B", "C", "D", "E") if k in case]
    a = (answer or "").strip().upper()
    if a in labels:
        return a
    m = re.search(r"\b([A-E])\b", a)
    return m.group(1) if m else (labels[0] if labels else "A")


def run(model, processor, case, imgs_dir):
    img_file = os.path.join(imgs_dir, os.path.basename(case["image_path"]))
    image = Image.open(img_file).convert("RGB")
    prompt = mcq_prompt(case) if case["question type"] == "mcq" else open_prompt(case)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    out = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return parse(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mcq-n", type=int, default=300)
    ap.add_argument("--open-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = json.load(open(args.json))["cases"]
    mcq = [c for c in cases if c.get("question type") == "mcq"]
    opn = [c for c in cases if c.get("question type") == "open-ended"]
    random.seed(args.seed)
    random.shuffle(mcq); random.shuffle(opn)
    mcq = mcq[:args.mcq_n]; opn = opn[:args.open_n]

    print(f"Loading model from {args.model} …", flush=True)
    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    # ---- MCQ accuracy ----
    correct, t0 = 0, time.time()
    for i, c in enumerate(mcq):
        _, ans = run(model, processor, c, args.imgs)
        pred = norm_mcq(ans, c)
        if pred == str(c.get("answer", "")).strip().upper():
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  MCQ {i+1}/{len(mcq)}  acc={correct/(i+1):.3f}", flush=True)
    acc = correct / len(mcq) if mcq else 0.0
    print(f"\n=== MCQ accuracy on {len(mcq)} train cases: {acc:.3f} ===\n", flush=True)

    # ---- open-ended: print for manual inspection ----
    print(f"=== {len(opn)} open-ended examples (reference vs model) ===", flush=True)
    for c in opn:
        reasoning, answer = run(model, processor, c, args.imgs)
        print("-" * 70)
        print("Q       :", c["question"][:120])
        print("REF ans :", str(c.get("answer", ""))[:160])
        print("MODEL   :", answer[:160])
        print("TRACE   :", reasoning[:200])
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
