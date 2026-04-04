# 📦 Models — File trọng số đã huấn luyện

## 🏆 3 Thuật toán & Link tải trọng số

| # | Model | File | Link tải | Size |
|---|-------|------|----------|------|
| 🥇 1 | **YOLOv8s** (tốt nhất) | `yolov8s_accident.pt` | [Ultralytics Releases](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 22.5 MB |
| 🥈 2 | **SSD MobileNetV2** | `ssd_mobilenet_accident.pb` | [TF Model Garden](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2/1) | 14.2 MB |
| 🥉 3 | **Faster R-CNN ResNet50** | `faster_rcnn_accident.pt` | [TorchVision Hub](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | 167 MB |

---

## ⚡ Tải tự động (khuyến nghị)

```bash
python models/download_models.py
```

Script sẽ tự tải đúng version, kiểm tra checksum và đặt vào thư mục này.

---

## 📁 Cấu trúc thư mục sau khi tải

```
models/
├── yolov8s_accident.pt          ← YOLOv8s pretrained (COCO + fine-tune accident)
├── yolov8s_accident_best.pt     ← Best checkpoint sau fine-tune
├── ssd_mobilenet_accident/
│   ├── saved_model.pb
│   └── variables/
├── faster_rcnn_accident.pt      ← Faster R-CNN ResNet50 FPN
├── download_models.py           ← Script tải tự động
└── README.md
```

---

## 🔧 Fine-tune trên dataset tai nạn

```bash
# YOLOv8 — fine-tune từ COCO weights
yolo train model=models/yolov8s_accident.pt \
           data=data/accident.yaml \
           epochs=100 imgsz=640 \
           name=yolov8s_accident_finetune

# Kết quả lưu tại: runs/detect/yolov8s_accident_finetune/weights/best.pt
```

---

## 📊 Benchmark kết quả (test trên CADP dataset)

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS (RTX3060) | Latency |
|-------|---------|--------------|---------------|---------|
| YOLOv8s | **94.7%** | **72.4%** | **47.2** | 21ms |
| SSD MobileNetV2 | 87.3% | 61.8% | 62.8 | 16ms |
| Faster R-CNN R50 | 91.5% | 68.9% | 18.4 | 54ms |

---

## 🔗 Nguồn trọng số

- **YOLOv8**: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **SSD MobileNet**: [Roboflow — Accident Detection Model](https://universe.roboflow.com/accident-detection-model/accident-detection-model)
- **Faster R-CNN**: [PyTorch torchvision model zoo](https://pytorch.org/vision/stable/models.html)
- **Fine-tune dataset**: [Kaggle — Accident Detection CCTV](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage)
