#!/usr/bin/env python3
"""
build_dino_index.py — Phase 1 of DINOv3 retrieval for MedReason.

Runs the FROZEN DINOv3 image encoder over all training images and builds a
searchable index of (embedding -> gold finding/answer). This index is later
used at inference to retrieve visually-similar training cases as in-context
examples, injecting medical specificity the base 7B model lacks.

DINOv3 is used as-is (no fine-tuning, no LoRA): we only run it forward to get
one embedding vector per image.

Outputs (to --out dir):
  dino_embeddings.npy   float32 [N, D]   L2-normalized image embeddings
  dino_meta.json        list[N] of {case_id, task_type, answer, question, image}

Usage (Isambard GPU node, after HF approval + `huggingface-cli login`):
  python build_dino_index.py \
    --json data/train/medreason_train_selection.json \
    --imgs data/train/imgs \
    --out  dino_index \
    --model facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch 32
"""
import argparse, json, os, numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def embed_images(model, processor, paths, device, batch):
    """Return L2-normalized CLS embeddings for a list of image paths."""
    vecs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        # pooler_output is the CLS token after pooling; fall back to mean of last_hidden_state
        if getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state[:, 0]   # CLS token
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        vecs.append(emb.cpu().numpy())
        if (i // batch) % 20 == 0:
            print(f"  embedded {i+len(chunk)}/{len(paths)}", flush=True)
    return np.concatenate(vecs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)  # for a quick smoke run
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = json.load(open(args.json))["cases"]
    if args.limit:
        cases = cases[:args.limit]

    # resolve image paths + keep only cases whose image exists
    rows, paths = [], []
    for c in cases:
        p = os.path.join(args.imgs, os.path.basename(c["image_path"]))
        if not os.path.exists(p):
            continue
        rows.append({
            "case_id": c["case_id"],
            "task_type": c.get("question type", c.get("task_type")),
            "answer": str(c.get("answer", "")).strip(),
            "question": str(c.get("question", "")).strip(),
            "image": os.path.basename(p),
        })
        paths.append(p)
    print(f"Embedding {len(paths)} training images with {args.model}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    print(f"DINOv3 loaded on {device}. embed dim will be inferred.", flush=True)

    emb = embed_images(model, processor, paths, device, args.batch)
    print(f"Embeddings shape: {emb.shape}", flush=True)

    np.save(os.path.join(args.out, "dino_embeddings.npy"), emb.astype(np.float32))
    with open(os.path.join(args.out, "dino_meta.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Saved index to {args.out}/ (dino_embeddings.npy + dino_meta.json)", flush=True)
    print("Index build complete. Next: retrieval-augmented inference + val measurement.", flush=True)


if __name__ == "__main__":
    main()
