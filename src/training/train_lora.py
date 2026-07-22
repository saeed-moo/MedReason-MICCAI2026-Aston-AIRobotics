#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tune Qwen2.5-VL on MedReason train data.

Plain PyTorch loop (robust across transformers versions). Trains a small LoRA
adapter; the base model stays frozen. Loss is computed ONLY on the target tokens
(the prompt + image tokens are masked out).

Quick smoke test first:
  python train_lora.py --data ft_data/train.jsonl --model <path> \
      --out lora_out --limit 50 --epochs 1 --batch 1

Full run:
  python train_lora.py --data ft_data/train.jsonl --model <path> \
      --out lora_out --epochs 1 --batch 4 --grad-accum 4
"""
import argparse, json, os, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


class MedReasonDataset(Dataset):
    def __init__(self, path, imgs_root, limit=0):
        self.rows = []
        with open(path) as f:
            for line in f:
                self.rows.append(json.loads(line))
        if limit:
            self.rows = self.rows[:limit]
        self.imgs_root = imgs_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(processor, imgs_root):
    def collate(batch):
        texts, images, prompt_texts = [], [], []
        for row in batch:
            # resolve image path (rows store paths relative to MedReason root)
            img_path = row["image"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(imgs_root, img_path)
            image = Image.open(img_path).convert("RGB")

            user_msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["prompt"]},
            ]}]
            # full = prompt + target ; prompt_only = prompt (for masking)
            prompt_only = processor.apply_chat_template(
                user_msg, tokenize=False, add_generation_prompt=True)
            full = prompt_only + row["target"]
            texts.append(full)
            prompt_texts.append(prompt_only)
            images.append(image)

        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True)
        # build labels: mask everything that belongs to the prompt
        labels = enc["input_ids"].clone()
        # compute prompt length per-sample by tokenizing the prompt alone
        prompt_enc = processor(text=prompt_texts, images=images,
                               return_tensors="pt", padding=True)
        for b in range(labels.shape[0]):
            plen = int(prompt_enc["attention_mask"][b].sum().item())
            labels[b, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("Loading base model…", flush=True)
    model = VLModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    ds = MedReasonDataset(args.data, args.imgs_root, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=make_collate(processor, args.imgs_root))
    print(f"Training on {len(ds)} examples, {len(dl)} batches/epoch", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  ep{epoch} step{step} loss={out.loss.item():.4f} "
                          f"({rate:.2f} batch/s)", flush=True)
                if step % args.save_every == 0:
                    model.save_pretrained(os.path.join(args.out, f"step{step}"))
                    print(f"  saved adapter at step {step}", flush=True)

    model.save_pretrained(args.out)
    print(f"Done. LoRA adapter saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
