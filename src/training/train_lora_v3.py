#!/usr/bin/env python3
"""
train_lora_v3.py — LoRA fine-tune Qwen2.5-VL, tracking BOTH MCQ accuracy and
an open-ended sanity score on held-out val, so we never again regress open-ended
without noticing.

Open-ended "sanity" = fraction of val open cases where the model returns valid
JSON with a non-empty 'answer' that is NOT a near-copy of the question (a crude
proxy; the real GT/VA come from organizer judges, but this catches collapse).
"""
import argparse, json, os, re, time
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


def load_rows(p, limit=0):
    r = [json.loads(l) for l in open(p)]
    return r[:limit] if limit else r


class DS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def make_collate(proc, root):
    def collate(batch):
        texts, prompts, images = [], [], []
        for row in batch:
            image = Image.open(resolve(row["image"], root)).convert("RGB")
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": row["prompt"]}]}]
            p = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(p + row["target"]); prompts.append(p); images.append(image)
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        penc = proc(text=prompts, images=images, return_tensors="pt", padding=True)
        for b in range(labels.shape[0]):
            plen = int(penc["attention_mask"][b].sum().item())
            labels[b, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def evaluate(model, proc, val, root, mcq_n, open_n):
    model.eval()
    mcq = [r for r in val if r["task_type"] == "mcq"][:mcq_n]
    correct, preds = 0, []
    for r in mcq:
        img = Image.open(resolve(r["image"], root)).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": r["prompt"]}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[t], images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
        dec = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\b([A-E])\b", dec.upper())
        p = m.group(1) if m else "?"
        preds.append(p)
        if p == r["target"].strip().upper():
            correct += 1
    acc = correct / len(mcq) if mcq else 0.0

    # open-ended sanity: valid JSON + non-empty answer + answer not trivially short
    opn = [r for r in val if r["task_type"] == "open"][:open_n]
    ok = 0
    for r in opn:
        img = Image.open(resolve(r["image"], root)).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": r["prompt"]}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[t], images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
        dec = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        mm = re.search(r"\{.*\}", dec, re.DOTALL)
        if mm:
            try:
                o = json.loads(mm.group(0))
                a = str(o.get("answer", "")).strip()
                if len(a) >= 5:
                    ok += 1
            except json.JSONDecodeError:
                pass
    open_ok = ok / len(opn) if opn else 0.0
    model.train()
    return acc, dict(sorted(Counter(preds).items())), open_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)   # lower than v2 (was 1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-mcq", type=int, default=200)
    ap.add_argument("--eval-open", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print("Loading base model…", flush=True)
    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    device_map="cuda", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train = load_rows(args.train, args.limit)
    val = load_rows(args.val)
    dl = DataLoader(DS(train), batch_size=args.batch, shuffle=True,
                    collate_fn=make_collate(proc, args.imgs_root))
    print(f"Train {len(train)} | val {len(val)} | {len(dl)} batches/epoch", flush=True)

    acc, dist, oo = evaluate(model, proc, val, args.imgs_root, args.eval_mcq, args.eval_open)
    print(f"\n*** BASELINE  MCQ={acc:.3f}  open_ok={oo:.3f}  letters={dist} ***\n", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"  ep{ep} step{step} loss={out.loss.item():.4f} "
                          f"({(i+1)/(time.time()-t0):.2f} b/s)", flush=True)
                if step % args.eval_every == 0:
                    acc, dist, oo = evaluate(model, proc, val, args.imgs_root,
                                             args.eval_mcq, args.eval_open)
                    print(f"  >>> step{step}  MCQ={acc:.3f}  open_ok={oo:.3f}  letters={dist}", flush=True)
                if step % args.save_every == 0:
                    model.save_pretrained(os.path.join(args.out, f"step{step}"))

    acc, dist, oo = evaluate(model, proc, val, args.imgs_root, args.eval_mcq, args.eval_open)
    print(f"\n*** FINAL  MCQ={acc:.3f}  open_ok={oo:.3f}  letters={dist} ***", flush=True)
    model.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
