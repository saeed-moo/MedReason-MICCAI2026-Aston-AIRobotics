#!/usr/bin/env python3
"""
train_lora_v2.py — LoRA fine-tune Qwen2.5-VL on MedReason with built-in
validation-accuracy tracking (MCQ exact-match on held-out val.jsonl).

Reports val MCQ accuracy BEFORE training and at intervals during, so we can
see whether fine-tuning actually helps and catch overfitting early.

Full run:
  python train_lora_v2.py \
    --train ft_data/train.jsonl --val ft_data/val.jsonl \
    --model <path> --out lora_medreason \
    --epochs 1 --grad-accum 16 --eval-n 200 --eval-every 400
"""
import argparse, json, os, re, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


def load_rows(path, limit=0):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


class DS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def make_collate(processor, root):
    def collate(batch):
        texts, prompts, images = [], [], []
        for row in batch:
            image = Image.open(resolve(row["image"], root)).convert("RGB")
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["prompt"]}]}]
            p = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(p + row["target"]); prompts.append(p); images.append(image)
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        penc = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        for b in range(labels.shape[0]):
            plen = int(penc["attention_mask"][b].sum().item())
            labels[b, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def eval_mcq_acc(model, processor, val_rows, root, n):
    """Generate on n MCQ val cases, score exact-match on the gold letter."""
    model.eval()
    mcq = [r for r in val_rows if r["task_type"] == "mcq"][:n]
    correct = 0
    for r in mcq:
        image = Image.open(resolve(r["image"], root)).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": r["prompt"]}]}]
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
        dec = processor.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\b([A-E])\b", dec.upper())
        pred = m.group(1) if m else "?"
        if pred == r["target"].strip().upper():
            correct += 1
    model.train()
    return correct / len(mcq) if mcq else 0.0


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
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("Loading base model…", flush=True)
    model = VLModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_rows = load_rows(args.train, args.limit)
    val_rows = load_rows(args.val)
    dl = DataLoader(DS(train_rows), batch_size=args.batch, shuffle=True,
                    collate_fn=make_collate(processor, args.imgs_root))
    print(f"Train {len(train_rows)} | val {len(val_rows)} | {len(dl)} batches/epoch", flush=True)

    # BASELINE val accuracy (before any training)
    base = eval_mcq_acc(model, processor, val_rows, args.imgs_root, args.eval_n)
    print(f"\n*** BASELINE val MCQ acc (pre-train): {base:.3f} ***\n", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    mcq_l, open_l = [], []
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(dl):
            tt = train_rows[i % len(train_rows)]  # approx; for logging split only
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  ep{epoch} step{step} loss={out.loss.item():.4f} ({rate:.2f} b/s)", flush=True)
                if step % args.eval_every == 0:
                    acc = eval_mcq_acc(model, processor, val_rows, args.imgs_root, args.eval_n)
                    print(f"  >>> step{step} val MCQ acc = {acc:.3f} (baseline was {base:.3f})", flush=True)
                if step % args.save_every == 0:
                    model.save_pretrained(os.path.join(args.out, f"step{step}"))

    final = eval_mcq_acc(model, processor, val_rows, args.imgs_root, args.eval_n)
    print(f"\n*** FINAL val MCQ acc: {final:.3f}  (baseline {base:.3f}, delta {final-base:+.3f}) ***", flush=True)
    model.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
