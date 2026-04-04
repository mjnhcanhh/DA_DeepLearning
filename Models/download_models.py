"""
download_models.py — Tự động tải trọng số 3 thuật toán
Chạy: python models/download_models.py
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ─── Thư mục lưu trọng số ────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent

# ─── Danh sách model cần tải ─────────────────────────────────────────────────
# YOLOv8s — pretrained COCO, dùng để fine-tune thêm với accident dataset
# SSD & Faster R-CNN — base weights từ PyTorch/TensorFlow model zoo
MODELS = [
    {
        "rank":     1,
        "name":     "YOLOv8s (Tốt nhất — pretrained COCO)",
        "filename": "yolov8s.pt",
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
        "size_mb":  22.5,
        "note":     "Dùng làm base, fine-tune thêm với CADP/Roboflow accident dataset",
    },
    {
        "rank":     2,
        "name":     "SSD MobileNetV2",
        "filename": "ssd_mobilenet_v2_coco.tar.gz",
        "url":      (
            "http://download.tensorflow.org/models/object_detection/tf2/"
            "20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
        ),
        "size_mb":  33.0,
        "note":     "TF2 SavedModel format; extract rồi dùng với TF Object Detection API",
    },
    {
        "rank":     3,
        "name":     "Faster R-CNN ResNet50 FPN (COCO)",
        "filename": "fasterrcnn_resnet50_fpn_coco.pth",
        "url":      "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "size_mb":  167.0,
        "note":     "PyTorch torchvision model zoo; load bằng torchvision.models.detection",
    },
]


# ─── Tqdm download hook ───────────────────────────────────────────────────────
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path, desc: str) -> bool:
    """Tải file với thanh tiến trình"""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                  miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  ❌ Lỗi tải: {e}")
        return False


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 60)
    print("  🚨 AI Accident Detection — Tải trọng số 3 thuật toán")
    print("═" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    success = []

    for m in MODELS:
        dest = MODELS_DIR / m["filename"]
        rank_icon = ["🏆", "🥈", "🥉"][m["rank"] - 1]

        print(f"\n{rank_icon} #{m['rank']} {m['name']}")
        print(f"   📁 File    : {m['filename']}")
        print(f"   🔗 URL     : {m['url'][:70]}...")
        print(f"   💾 Size    : ~{m['size_mb']} MB")
        print(f"   📝 Ghi chú : {m['note']}")

        if dest.exists():
            actual_mb = file_size_mb(dest)
            print(f"   ✅ Đã có sẵn ({actual_mb:.1f} MB) — bỏ qua tải lại")
            success.append(m["filename"])
            continue

        print(f"   ⬇️  Đang tải...")
        ok = download_file(m["url"], dest, m["filename"])
        if ok and dest.exists():
            actual_mb = file_size_mb(dest)
            print(f"   ✅ Tải xong! ({actual_mb:.1f} MB)")
            success.append(m["filename"])
        else:
            print(f"   ⚠️  Tải thất bại — xem hướng dẫn thủ công bên dưới")

    # ── Tóm tắt ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  ✅ Thành công: {len(success)}/{len(MODELS)} model")

    existing = [f for f in MODELS_DIR.iterdir()
                if f.suffix in ('.pt', '.pth', '.pb', '.gz', '.h5')]
    if existing:
        print(f"\n  📦 Files trong models/:")
        for f in sorted(existing):
            print(f"     {f.name:45s}  {file_size_mb(f):.1f} MB")

    print("\n  📌 Bước tiếp theo:")
    print("     1. Fine-tune YOLOv8 với accident dataset:")
    print("        yolo train model=models/yolov8s.pt data=data/accident.yaml epochs=100")
    print("     2. Chạy dashboard:")
    print("        streamlit run app.py")
    print("─" * 60 + "\n")


# ─── Kiểm tra nhanh model load ────────────────────────────────────────────────
def verify_models():
    """Kiểm tra có thể load model hay không"""
    print("\n🔍 Kiểm tra model...")

    # YOLOv8
    yolo_path = MODELS_DIR / "yolov8s.pt"
    if yolo_path.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(yolo_path))
            print(f"  ✅ YOLOv8s load OK — {model.info(verbose=False)}")
        except ImportError:
            print("  ⚠️  ultralytics chưa cài: pip install ultralytics")
        except Exception as e:
            print(f"  ❌ YOLOv8 lỗi: {e}")

    # Faster R-CNN
    rcnn_path = MODELS_DIR / "fasterrcnn_resnet50_fpn_coco.pth"
    if rcnn_path.exists():
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            model = fasterrcnn_resnet50_fpn(pretrained=False)
            state = torch.load(str(rcnn_path), map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            print("  ✅ Faster R-CNN load OK")
        except ImportError:
            print("  ⚠️  torchvision chưa cài: pip install torchvision")
        except Exception as e:
            print(f"  ❌ Faster R-CNN lỗi: {e}")

    print("  ℹ️  SSD MobileNet cần TensorFlow 2.x để verify")


if __name__ == "__main__":
    main()
    if "--verify" in sys.argv:
        verify_models()
