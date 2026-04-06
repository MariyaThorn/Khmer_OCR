"""
predict.py — Run inference with a trained Khmer OCR model
─────────────────────────────────────────────────────────
Usage (single image):
    python predict.py \
        --checkpoint outputs/best_model.pth \
        --image path/to/line_image.png

Usage (folder of images):
    python predict.py \
        --checkpoint outputs/best_model.pth \
        --image_dir  path/to/images/

Usage (from a parquet file, no labels needed):
    python predict.py \
        --checkpoint  outputs/best_model.pth \
        --parquet     data/unlabeled.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import CHAR2IDX, IDX2CHAR, NUM_CLASSES, ctc_decode
from data.dataset import get_transforms, resize_to_height
from models.crnn import KhmerOCR


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("args", {})
    img_height = saved.get("img_height", 32)
    rnn_hidden  = saved.get("rnn_hidden", 256)
    rnn_layers  = saved.get("rnn_layers", 2)

    model = KhmerOCR(NUM_CLASSES, rnn_hidden, rnn_layers).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, img_height


# ── Predict one image ─────────────────────────────────────────────────────────

def predict_image(model, img: Image.Image, img_height: int, device: torch.device) -> str:
    transform = get_transforms(img_height, augment=False)
    img = resize_to_height(img.convert("RGB"), img_height)
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 1, H, W)

    with torch.no_grad():
        logits = model(tensor)                        # (T, 1, C)
        indices = logits.argmax(dim=2).squeeze(1)     # (T,)
        text = ctc_decode(indices.tolist())

    return text


# ── Predict from parquet ──────────────────────────────────────────────────────

def predict_from_parquet(model, parquet_path: str, img_height: int, device: torch.device):
    import io
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    results = []

    for i, row in df.iterrows():
        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            img = Image.open(io.BytesIO(img_data["bytes"]))
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
        else:
            img = img_data

        pred = predict_image(model, img, img_height, device)
        results.append(pred)
        print(f"[{i}] {pred}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Khmer OCR Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image",      help="Path to a single image file")
    p.add_argument("--image_dir",  help="Directory of image files")
    p.add_argument("--parquet",    help="Parquet file with image column")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_height = load_model(args.checkpoint, device)
    print(f"Model loaded | img_height={img_height} | device={device}\n")

    if args.image:
        img  = Image.open(args.image)
        pred = predict_image(model, img, img_height, device)
        print(f"Prediction: {pred}")

    elif args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        paths = [p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts]
        paths.sort()
        for path in paths:
            img  = Image.open(path)
            pred = predict_image(model, img, img_height, device)
            print(f"{path.name}: {pred}")

    elif args.parquet:
        predict_from_parquet(model, args.parquet, img_height, device)

    else:
        print("Please provide --image, --image_dir, or --parquet")


if __name__ == "__main__":
    main()
