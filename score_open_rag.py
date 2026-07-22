#!/usr/bin/env python3
"""
score_open_rag.py — measure DINOv3 retrieval-augmented open-ended answering on
the val set, head-to-head against the no-retrieval baseline, using the SAME
proxies (token F1 + embedding cosine) as score_open.py so numbers are comparable.

For each val open-ended case:
  1. embed the val image with DINOv3 (same model as the index)
  2. find the k nearest TRAIN images by cosine similarity over the prebuilt index
  3. inject their gold findings into the Qwen prompt as reference context
  4. generate, score answer vs gold

A retrieval-distance GATE is included: if the nearest neighbor is too dissimilar
(below --sim-floor), we DROP the retrieved context for that case (likely OOD /
no useful match), falling back to plain reasoning. This prevents misleading
neighbors from hurting.

Reports mean F1 + cosine WITH retrieval; compare to your baseline run.

Usage (Isambard GPU):
  python score_open_rag.py \
    --val ft_data_v3/val.jsonl \
    --model models/Qwen2.5-VL-merged \
    --index dino_index \
    --dino facebook/dinov3-vitl16-pretrain-lvd1689m \
    --k 3 --sim-floor 0.5 --n 202
"""
import argparse, json, os, re, string
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, AutoModel
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

OPEN_SYSTEM = ("You are an expert radiologist answering an open-ended question about a "
               "medical image. First identify the modality and region you can see, then give "
               "a precise, image-grounded answer.")


def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return s.split()


def token_f1(pred, gold):
    from collections import Counter
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return str(json.loads(m.group(0)).get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    return raw.strip()


def resolve(img, root):
    return img if os.path.isabs(img) else os.path.join(root, img)


def build_prompt(question, retrieved):
    """retrieved: list of gold finding strings (may be empty)."""
    ctx = ""
    if retrieved:
        lines = "\n".join(f"- {r}" for r in retrieved)
        ctx = ("Findings reported in visually similar reference cases (use only if "
               "consistent with what you actually see; ignore if not relevant):\n"
               f"{lines}\n\n")
    return (f"{OPEN_SYSTEM}\n\n{ctx}Question: {question}\n\n"
            'Respond ONLY as compact JSON with keys "reasoning_trace" and "answer".\n'
            '- "reasoning_trace": 2-3 sentences describing ONLY findings visible in the image.\n'
            '- "answer": a single specific, committed clinical statement.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--dino", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--imgs-root", default=".")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sim-floor", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=202)
    args = ap.parse_args()

    # load index
    emb = np.load(os.path.join(args.index, "dino_embeddings.npy"))   # [N, D] normalized
    meta = json.load(open(os.path.join(args.index, "dino_meta.json")))
    print(f"Index: {emb.shape[0]} train images, dim {emb.shape[1]}", flush=True)

    rows = [json.loads(l) for l in open(args.val) if json.loads(l)["task_type"] == "open"][:args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DINOv3 for query embedding
    dino_proc = AutoImageProcessor.from_pretrained(args.dino)
    dino = AutoModel.from_pretrained(args.dino, dtype=torch.bfloat16).to(device).eval()
    # Qwen generator
    qproc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    qwen = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                   device_map="cuda", trust_remote_code=True).eval()

    def dino_embed(pil):
        inp = dino_proc(images=[pil], return_tensors="pt").to(device)
        with torch.inference_mode():
            o = dino(**inp)
        v = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        return v.cpu().numpy()[0]

    f1s, sims_used, n_with_ctx = [], [], 0
    try:
        from sentence_transformers import SentenceTransformer, util
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sbert = None
    cos_scores = []

    for i, r in enumerate(rows):
        img = Image.open(resolve(r["image"], args.imgs_root)).convert("RGB")
        q = dino_embed(img)                      # [D]
        sims = emb @ q                           # cosine (both normalized) [N]
        top = np.argsort(-sims)[:args.k]
        best = float(sims[top[0]])
        retrieved = []
        if best >= args.sim_floor:
            retrieved = [meta[j]["answer"] for j in top if meta[j]["answer"]]
            n_with_ctx += 1
        sims_used.append(best)

        prompt = build_prompt(r["prompt"].split("Question:")[-1].strip() if "Question:" in r["prompt"] else r["prompt"], retrieved)
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": prompt}]}]
        text = qproc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = qproc(text=[text], images=[img], return_tensors="pt").to(qwen.device)
        with torch.inference_mode():
            out = qwen.generate(**enc, max_new_tokens=200, do_sample=False)
        dec = qproc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(dec)
        gold = json.loads(r["target"])["answer"] if r["target"].startswith("{") else r["target"]
        f1s.append(token_f1(pred, gold))
        if sbert is not None:
            e = sbert.encode([pred, gold], convert_to_tensor=True)
            cos_scores.append(float(util.cos_sim(e[0], e[1])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}  F1={np.mean(f1s):.3f}  ctx_used={n_with_ctx}", flush=True)

    print(f"\n=== DINOv3-RAG on {len(rows)} open cases (k={args.k}, sim-floor={args.sim_floor}) ===")
    print(f"mean token F1:      {np.mean(f1s):.3f}")
    if cos_scores:
        print(f"mean embed cos sim: {np.mean(cos_scores):.3f}")
    print(f"cases with retrieved context: {n_with_ctx}/{len(rows)}")
    print(f"mean top-1 retrieval similarity: {np.mean(sims_used):.3f}")
    print("\nCompare to baseline (no retrieval): F1 0.173, cos 0.363")


if __name__ == "__main__":
    main()
