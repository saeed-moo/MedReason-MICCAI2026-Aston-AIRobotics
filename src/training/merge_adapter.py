#!/usr/bin/env python3
"""
merge_adapter.py — merge the v4 LoRA adapter into the base Qwen2.5-VL weights,
producing a standalone merged checkpoint. A merged model loads like any normal
model (no peft at runtime), which avoids the adapter-attach KeyError seen when
the model is memory-offloaded. Then runs a quick real inference to confirm it works.

Usage (on Isambard GPU node):
  python merge_adapter.py \
    --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
    --adapter lora_medreason_v4 \
    --out models/Qwen2.5-VL-merged \
    --test-image data/train/imgs/<some.png>
"""
import argparse, os, torch
from transformers import AutoProcessor
from peft import PeftModel
from PIL import Image
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-image", default=None)
    args = ap.parse_args()

    print("Loading base model (full precision, single device)…", flush=True)
    # IMPORTANT: load fully onto GPU (no device_map='auto' offload) so merge is clean
    model = VLModel.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)

    print(f"Attaching adapter {args.adapter}…", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base weights…", flush=True)
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.out}…", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    processor.save_pretrained(args.out)
    print("Merged checkpoint saved.", flush=True)

    # quick real inference sanity check
    if args.test_image and os.path.exists(args.test_image):
        print("\nRunning a real inference to confirm the merged model generates…", flush=True)
        img = Image.open(args.test_image).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe the imaging modality and main visible structure in one sentence."}]}]
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False)
        dec = processor.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print("MODEL OUTPUT:", dec.strip(), flush=True)
        print("\nIf the output above is a coherent sentence, the merged model works.", flush=True)


if __name__ == "__main__":
    main()
