#!/usr/bin/env python3
"""
test_hybrid_container.py — run the HYBRID container's inference logic on Isambard's
GPU against test/cases.json, proving the full pipeline (load base + adapter, toggle
per task, generate, write results.json) works end-to-end before Docker submission.

This mirrors what /opt/app/process.py does inside the container, but runs directly
in the Isambard conda env (GPU available) so we validate the adapter-toggle path
that the no-GPU laptop cannot exercise.

Usage (Isambard GPU node):
  python test_hybrid_container.py \
    --cases MedReason-Challenge-Docker/MedReason-Docker/test/cases.json \
    --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
    --adapter lora_medreason_v4 \
    --imgs-root MedReason-Challenge-Docker/MedReason-Docker/test \
    --out /tmp/hybrid_test_results.json
"""
import argparse, json, os, re
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


def mcq_prompt(case):
    opts = case.get("options", [])
    lines = "\n".join(f"{o['label']}. {o['text']}" for o in opts)
    return (f"{MCQ_SYSTEM}\n\nQuestion: {case['question']}\n\nOptions:\n{lines}\n\n"
            "Answer with ONLY the single letter of the best option (A, B, C, D, or E). "
            "Output just the letter, nothing else.")


def open_prompt(case):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {case['question']}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image.\n'
            '- "answer": a single specific, committed clinical statement.')


def load_cases(path):
    data = json.load(open(path))
    return data["cases"] if isinstance(data, dict) and "cases" in data else data


def image_paths(case, root):
    paths = case.get("image_paths") or ([case["image_path"]] if case.get("image_path") else [])
    out = []
    for p in paths:
        cand = p if os.path.isabs(p) else os.path.join(root, p)
        if not os.path.exists(cand):
            cand = os.path.join(root, os.path.basename(p))
        out.append(cand)
    return out


def normalize_mcq(text, case):
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    labels = [o["label"] for o in case.get("options", [])]
    return labels[0] if labels else "A"


def parse_open(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            return str(o.get("reasoning_trace", "")).strip(), str(o.get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    return text.strip(), text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--imgs-root", required=True)
    ap.add_argument("--out", default="/tmp/hybrid_test_results.json")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} cases", flush=True)

    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    print("Loading base + adapter on GPU…", flush=True)
    model = VLModel.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                    trust_remote_code=True).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="ft")
    model.eval()
    print("Loaded. Adapter attached as 'ft'.", flush=True)

    def gen(imgs, prompt, max_new):
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        msg = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=imgs if imgs else None, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        return proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    answers = []
    for i, case in enumerate(cases):
        imgs = [Image.open(p).convert("RGB") for p in image_paths(case, args.imgs_root)]
        tt = case.get("task_type", "mcq")
        if tt == "mcq":
            model.set_adapter("ft")                       # adapter ON
            raw = gen(imgs, mcq_prompt(case), 8)
            ans = normalize_mcq(raw, case)
            trace = "Selected the option best supported by the visible image evidence."
            print(f"  case {i+1} [MCQ, adapter ON]  raw={raw!r} -> {ans}", flush=True)
        else:
            with model.disable_adapter():                 # adapter OFF (base)
                raw = gen(imgs, open_prompt(case), 512)
            trace, ans = parse_open(raw)
            print(f"  case {i+1} [OPEN, adapter OFF] answer={ans[:60]!r} trace_words={len(trace.split())}", flush=True)
        answers.append({"case_id": case["case_id"], "task_type": tt,
                        "answer": ans, "reasoning_trace": trace})

    out = {"name": "MedReason hybrid predictions", "type": "Medical visual reasoning",
           "answers": answers, "version": {"major": 1, "minor": 0}}
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {len(answers)} predictions to {args.out}", flush=True)
    print("If MCQ returned a letter and OPEN returned a JSON answer with a multi-word", flush=True)
    print("trace, the hybrid container logic works on GPU end-to-end.", flush=True)


if __name__ == "__main__":
    main()
