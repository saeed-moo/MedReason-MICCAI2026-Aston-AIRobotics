  #!/usr/bin/env python3
"""
medreason_infer.py — run Qwen2.5-VL over a MedReason split and write results.json.

Reads <data_dir>/<split>.json (the participant-facing JSON with a 'cases' list)
and the images in <data_dir>/imgs/, then writes results.json in the format the
MedReason evaluator expects:
  - MCQ        -> {"case_id", "task_type":"mcq",  "answer": "<A-E>"}
  - open-ended -> {"case_id", "task_type":"open", "reasoning_trace": "...",
                                                   "answer": "..."}

Example:
  python medreason_infer.py \
      --json   /scratch/.../data/validation/medreason_validation_participant_facing.json \
      --imgs   /scratch/.../data/validation/imgs \
      --model  /scratch/.../models/Qwen2.5-VL \
      --out    /scratch/.../outputs/results.json \
      --limit  8          # optional: only first N cases, for a quick smoke test
"""
import argparse, json, os, re, sys, time

import torch
from transformers import AutoProcessor
from PIL import Image

# Qwen2.5-VL class import is version-dependent; try the known locations.
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as VLModel
    except ImportError:
        sys.exit("Could not import a Qwen2.5-VL model class from transformers. "
                 "Check `pip show transformers` and the model card for the right class.")


# ---------- prompt templates ------------------------------------------------
# Kept deliberately grounded: the open-ended trace must describe only what is
# visible, because the evaluator caps the visual-accuracy score when the
# reasoning trace is unfaithful to the image.
MCQ_INSTRUCTION = (
    "You are an expert radiologist. Look at the medical image and answer the "
    "multiple-choice question. Respond with ONLY the single letter (A, B, C, "
    "D, or E) of the best option. Do not explain."
)
OPEN_INSTRUCTION = (
    "You are an expert radiologist. Look at the medical image and answer the "
    "question. First give a brief reasoning that refers ONLY to findings "
    "actually visible in the image, then give a short final answer.\n"
    "Format exactly as:\n"
    "REASONING: <2-4 sentences grounded in the image>\n"
    "ANSWER: <concise final answer>"
)


def build_messages(case, img_path):
    qtype = case.get("question type")
    if qtype == "mcq":
        opts = "\n".join(case[k] for k in ("A", "B", "C", "D", "E") if k in case)
        text = f"{MCQ_INSTRUCTION}\n\nQuestion: {case['question']}\n\nOptions:\n{opts}"
    else:
        text = f"{OPEN_INSTRUCTION}\n\nQuestion: {case['question']}"
    return [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": text},
    ]}]


def parse_mcq(text):
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else "A"          # fallback so every case gets a label


def parse_open(text):
    reasoning, answer = "", text.strip()
    rm = re.search(r"REASONING:\s*(.*?)\s*ANSWER:", text, re.S | re.I)
    am = re.search(r"ANSWER:\s*(.*)", text, re.S | re.I)
    if rm:
        reasoning = rm.group(1).strip()
    if am:
        answer = am.group(1).strip()
    if not reasoning:                        # never leave the trace empty
        reasoning = answer
    return reasoning, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",  required=True)
    ap.add_argument("--imgs",  required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    cases = json.load(open(args.json))["cases"]
    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases.", flush=True)

    print("Loading model…", flush=True)
    model = VLModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    results, t0 = [], time.time()
    for i, case in enumerate(cases):
        img_file = os.path.join(args.imgs, os.path.basename(case["image_path"]))
        try:
            image = Image.open(img_file).convert("RGB")
        except Exception as e:
            print(f"[warn] {case['case_id']}: cannot open {img_file}: {e}", flush=True)
            image = None

        messages = build_messages(case, img_file)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image] if image else None,
                           return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        trimmed = gen[0][inputs["input_ids"].shape[1]:]
        text = processor.decode(trimmed, skip_special_tokens=True).strip()

        if case.get("question type") == "mcq":
            results.append({"case_id": case["case_id"], "task_type": "mcq",
                            "answer": parse_mcq(text)})
        else:
            reasoning, answer = parse_open(text)
            results.append({"case_id": case["case_id"], "task_type": "open",
                            "reasoning_trace": reasoning, "answer": answer})

        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(cases)}  ({rate:.2f} cases/s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"name": "MedReason predictions",
           "type": "Medical visual reasoning",
           "answers": results,
           "version": {"major": 1, "minor": 0}}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {len(results)} predictions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
