import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
import sys
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from model import get_model

IMG_SIZE = 640
SCORE_THRESH = 0.6
MAX_DET = 10
CLASS_NAMES = ["background", "accident"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def preprocess(img_path):
    """Read image -> resize -> tensor [0,1]. Do not ImageNet-normalize."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
    return tensor, img_np


def load_model(weights_path, device):
    model = get_model(num_classes=len(CLASS_NAMES))
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def draw_predictions(img_np, output, out_path):
    img = Image.fromarray((img_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    labels = output["labels"].detach().cpu()

    kept = torch.where(scores >= SCORE_THRESH)[0][:MAX_DET]
    for idx in kept:
        box = boxes[idx].tolist()
        score = float(scores[idx])
        label = int(labels[idx])
        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, max(0, box[1] - 16)), f"{name} {score:.2f}", fill="red")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return len(kept)


@torch.no_grad()
def run(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(args.weights, device)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    else:
        images = [input_path]

    out_dir = Path(args.output)
    for img_path in images:
        tensor, img_np = preprocess(img_path)
        output = model([tensor.to(device)])[0]
        out_path = out_dir / f"result_{img_path.stem}.jpg"
        n = draw_predictions(img_np, output, out_path)
        print(f"{img_path} -> {out_path} ({n} detections)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Faster R-CNN accident inference.")
    parser.add_argument("--input", required=True, help="Image path or directory.")
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "outputs" / "weights_clean_fixed" / "best_model.pth"),
        help="Path to best_model.pth or a raw state_dict checkpoint.",
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "inference_results"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
