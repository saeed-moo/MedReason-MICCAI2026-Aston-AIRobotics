#!/usr/bin/env python3
"""
gen_open_answers.py — generate open-ended answers ONCE and SAVE them.

Why separate from scoring: generation is the expensive part. Saving the answers
lets us judge them with different metrics/judges, and read them for error
analysis, without regenerating every time.

Works for any config we want to compare:
  base            : --adapter (omit)
  MCQ adapter     : --adapter lora_medreason_v4
  open-only LoRA  : --adapter lora_open

Output JSONL, one row per case:
  {case_id, question, gold, pred_answer, pred_trace, trace_words, answer_words}

Usage (Isambard GPU):
  python gen_open_answers.py --val ft_data_open/val.jsonl \
    --model MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
    --out answers_base.jsonl --n 202
"""
import argparse, json, os, re
import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. Examine the image carefully and answer with clinical precision.")


def get_question(prompt):
    if "Question:" in prompt:
        a = prompt.split("Question:", 1)[1]
        for m in ("\n\nOptions:", "\n\nRespond ONLY", "\n\nAnswer with",
                  "Respond ONLY", "Answer with"):
            if m in a:
                a = a.split(m)[0]
        return a.strip()
    return prompt.strip()


def open_prompt(q):
    return (f"{OPEN_SYSTEM}\n\nQuestion: {q}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image '
            "(modality, anatomical location, shape, margins, density or signal intensity). "
            "Do NOT state anything you cannot see.\n"
            '- "answer": a single specific, committed clinical statement that directly answers '
            "the question.")


def extract(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            return (str(o.get("answer", "")).strip(),
                    str(o.get("reasoning_trace", "")).strip())
        except json.JSONDecodeError:
            pass
    return raw.strip(), raw.strip()


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None,
                    help="optional LoRA adapter dir; omit for pure base model")
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=202)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val)
            if json.loads(l)["task_type"] == "open"][:args.n]
    print(f"{len(rows)} open-ended cases", flush=True)

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    trust_remote_code=True).to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"adapter attached: {args.adapter}", flush=True)
    else:
        print("no adapter — pure base model", flush=True)
    model.eval()

    out = open(args.out, "w", encoding="utf-8")
    for i, r in enumerate(rows):
        img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
        q = get_question(r["prompt"])
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": open_prompt(q)}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
        dec = proc.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ans, trace = extract(dec)
        gold = (json.loads(r["target"])["answer"]
                if r["target"].strip().startswith("{") else r["target"])
        out.write(json.dumps({
            "case_id": r.get("case_id", f"row{i}"),
            "image": r["image"],
            "question": q,
            "gold": str(gold).strip(),
            "pred_answer": ans,
            "pred_trace": trace,
            "answer_words": len(ans.split()),
            "trace_words": len(trace.split()),
        }, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    out.close()
    print(f"Saved answers to {args.out}", flush=True)


if __name__ == "__main__":
    main()

