#!/usr/bin/env python3
"""
build_dino_index_trainonly.py — build the DINOv3 index from the TRAIN split only
(ft_data_v3/train.jsonl), excluding val images, so retrieval evaluation on val is
HONEST (no leakage / self-retrieval).

Usage:
  python build_dino_index_trainonly.py \
    --train ft_data_v3/train.jsonl --imgs-root . \
    --out dino_index_trainonly \
    --model facebook/dinov3-vitl16-pretrain-lvd1689m --batch 32
"""
import argparse, json, os, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seen = set()
    rows, paths = [], []
    for line in open(args.train):
        r = json.loads(line)
        if r["task_type"] != "open" and r["task_type"] != "mcq":
            continue
        img = r["image"] if os.path.isabs(r["image"]) else os.path.join(args.imgs_root, r["image"])
        # train.jsonl may contain duplicates (open upweighting) -> dedupe by image
        key = os.path.basename(img)
        if key in seen or not os.path.exists(img):
            continue
        seen.add(key)
        gold = r["target"]
        if gold.startswith("{"):
            try:
                gold = json.loads(gold).get("answer", "")
            except json.JSONDecodeError:
                pass
        rows.append({"case_id": r.get("case_id", key), "task_type": r["task_type"],
                     "answer": str(gold).strip(), "image": key})
        paths.append(img)

    print(f"Train-only index: {len(paths)} unique images", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    vecs = []
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        with torch.inference_mode():
            o = model(**inp)
        v = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        vecs.append(v.cpu().numpy())
        if (i // args.batch) % 20 == 0:
            print(f"  embedded {i+len(chunk)}/{len(paths)}", flush=True)
    emb = np.concatenate(vecs, 0).astype(np.float32)
    np.save(os.path.join(args.out, "dino_embeddings.npy"), emb)
    json.dump(rows, open(os.path.join(args.out, "dino_meta.json"), "w"), ensure_ascii=False)
    print(f"Saved {emb.shape} to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
