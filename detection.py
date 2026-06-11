# -*- coding: utf-8 -*-
"""
detection.py — Module nhận diện vật thể & tai nạn
Hỗ trợ 3 thuật toán: SSD , YOLOv12, Faster R-CNN
"""

import cv2
import base64
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

class Algorithm(Enum):
    SSD         = "SSD"
    YOLOV12     = "YOLOv12"
    FASTER_RCNN = "Faster R-CNN"


@dataclass
class Detection:
    """Kết quả nhận diện một đối tượng"""
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2
    label:      str
    confidence: float
    algorithm:  str
    timestamp:  float = field(default_factory=time.time)

    @property
    def is_accident(self) -> bool:
        return self.label.lower() in ("accident", "crash", "collision")

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class FrameResult:
    """Kết quả xử lý một frame"""
    detections: List[Detection]
    fps:        float
    latency_ms: float
    algorithm:  str
    frame_id:   int

    @property
    def accident_detected(self) -> bool:
        return any(d.is_accident for d in self.detections)

    @property
    def accident_confidence(self) -> float:
        accs = [d.confidence for d in self.detections if d.is_accident]
        return max(accs) if accs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ALGORITHM_CONFIG = {
    Algorithm.SSD: {
        "map50":         None,
        "map50_95":      None,
        "precision":     None,
        "recall":        None,
        "f1":            None,
        "fps_mean":      None,
        "fps_std":       None,
        "latency_mean":  None,
        "latency_std":   None,
        "model_size_mb": 22.0,
        "description":   "SSD - mo hinh phat hien tai nan giao thong thoi gian thuc",
        "rank":          3,
    },
    Algorithm.YOLOV12: {
        "map50":         None,
        "map50_95":      None,
        "precision":     None,
        "recall":        None,
        "f1":            None,
        "fps_mean":      None,
        "fps_std":       None,
        "latency_mean":  None,
        "latency_std":   None,
        "model_size_mb": 6.0,
        "description":   "YOLOv12 - mo hinh phat hien tai nan thoi gian thuc",
        "rank":          1,
    },
    Algorithm.FASTER_RCNN: {
        "map50":         0.638,
        "map50_95":      None,
        "precision":     None,
        "recall":        None,
        "f1":            None,
        "fps_mean":      7.4,
        "fps_std":       1.0,
        "latency_mean":  135.0,
        "latency_std":   10.0,
        "model_size_mb": 167.0,
        "description":   "Faster R-CNN - mo hinh da huan luyen cho phat hien tai nan",
        "rank":          2,
    },
}

CLASSES = ["accident"]
CLASS_NAMES = {0: "background", 1: "accident"}

# Màu sắc cho từng class (BGR)
CLASS_COLORS = {
    "accident":  (0,   0,   255),   # đỏ
    "crash":     (0,   0,   255),
    "collision": (0,   0,   255),
    "car":       (0,   255, 0  ),   # xanh lá
    "motorbike": (0,   200, 50 ),
    "truck":     (255, 128, 0  ),   # cam
    "person":    (255, 255, 0  ),   # vàng
    "bicycle":   (0,   255, 255),   # cyan
    "bus":       (128, 0,   255),   # tím
}
DEFAULT_COLOR = (200, 200, 200)


# ══════════════════════════════════════════════════════════════════════════════
# CORE HELPER FUNCTIONS (dùng bởi app.py)
# ══════════════════════════════════════════════════════════════════════════════

def classify_label(label: str) -> int:
    """
    Phân loại mức độ nguy hiểm của nhãn.
    Trả về: 0 = bình thường, 1 = cảnh báo, 2 = tai nạn
    """
    label = label.lower().strip()
    if label in ("accident", "crash", "collision"):
        return 2
    if label in ("fire", "smoke", "warning", "danger"):
        return 1
    return 0


def to_b64(frame: np.ndarray) -> str:
    """Chuyển frame OpenCV sang base64 JPEG string có prefix data URI"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def draw_boxes(
    frame: np.ndarray,
    results,
    conf_thr: float,
    algo: str,
    cfg: dict = None,
) -> Tuple[np.ndarray, list, int, Optional[float]]:
    """
    Ve bounding box len frame tu ket qua Ultralytics (SSD/YOLOv12).

    Args:
        frame    : ảnh gốc (BGR numpy array)
        results  : danh sách kết quả từ model.predict()
        conf_thr : ngưỡng confidence tối thiểu
        algo     : tên thuật toán (để ghi vào detection dict)

    Returns:
        ann      : frame đã vẽ box
        det      : list[dict] các detection
        level    : mức độ nguy hiểm cao nhất trong frame (0/1/2)
        max_conf : confidence cao nhất, hoặc None nếu không có detection
    """
    ann      = frame.copy()
    det      = []
    level    = 0
    max_conf = 0.0
    cfg = cfg or {}
    score_thr = max(float(conf_thr), float(cfg.get("score_thresh", 0.0) or 0.0))
    max_det = int(cfg.get("max_det", 100) or 100)
    allowed_class_ids = set(cfg.get("allowed_class_ids", []))
    allowed_labels = {str(x).lower() for x in cfg.get("allowed_labels", [])}

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < score_thr:
                continue

            cls_id = int(box.cls[0])
            label  = r.names.get(cls_id, str(cls_id))
            if allowed_class_ids and cls_id not in allowed_class_ids:
                continue
            if allowed_labels and label.lower() not in allowed_labels:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            lv    = classify_label(label)
            level = max(level, lv)
            if conf > max_conf:
                max_conf = conf

            # Chọn màu
            color = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)
            if lv == 2:
                color = (0, 0, 255)     # đỏ override nếu là tai nạn
            elif lv == 1:
                color = (0, 165, 255)   # cam nếu cảnh báo

            # Vẽ box + label
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            text  = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(ann, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(ann, text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            det.append({
                "label": label,
                "conf":  round(conf, 3),
                "bbox":  [x1, y1, x2, y2],
                "level": lv,
                "algo":  algo,
            })
            if len(det) >= max_det:
                return ann, det, level, (max_conf if max_conf > 0 else None)

    return ann, det, level, (max_conf if max_conf > 0 else None)


def run_faster_rcnn(
    model,
    frame: np.ndarray,
    cfg:   dict,
    conf_thr: float,
) -> Tuple[np.ndarray, list, int, Optional[float]]:
    """
    Chạy inference Faster R-CNN (PyTorch / torchvision).

    Args:
        model    : model đã load (torchvision detection model)
        frame    : ảnh gốc (BGR numpy array)
        cfg      : ALGO_CONFIG[algo] — cần có key 'labels' (dict {id: name})
        conf_thr : ngưỡng confidence tối thiểu

    Returns:
        ann, det, level, max_conf  (giống draw_boxes)
    """
    import torch
    from torchvision.ops import nms as torch_nms
    import torchvision.transforms.functional as F

    ann      = frame.copy()
    det      = []
    level    = 0
    max_conf = 0.0

    # Chuyển BGR → RGB → tensor
    orig_h, orig_w = frame.shape[:2]
    input_size = int(cfg.get("input_size", 0) or 0)
    infer_frame = frame
    scale_x = scale_y = 1.0
    if input_size > 0:
        infer_frame = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        scale_x = orig_w / float(input_size)
        scale_y = orig_h / float(input_size)

    rgb    = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model([tensor])[0]

    # Lấy class_names từ config (key trong models.py là "class_names")
    labels_map: dict = cfg.get("class_names", cfg.get("labels", {}))

    boxes_t  = outputs.get("boxes", torch.empty((0, 4), device=device))
    labels_t = outputs.get("labels", torch.empty((0,), dtype=torch.long, device=device))
    scores_t = outputs.get("scores", torch.empty((0,), device=device))

    score_thr = max(float(conf_thr), float(cfg.get("score_thresh", 0.0) or 0.0))
    max_det = int(cfg.get("max_det", 100) or 100)
    nms_thr = float(cfg.get("nms_thresh", 0.45) or 0.45)
    allowed_class_ids = set(cfg.get("allowed_class_ids", []))

    keep = (scores_t >= score_thr) & (labels_t != 0)
    if allowed_class_ids:
        allowed_mask = torch.zeros_like(keep, dtype=torch.bool)
        for class_id in allowed_class_ids:
            allowed_mask |= labels_t == int(class_id)
        keep &= allowed_mask
    boxes_t, labels_t, scores_t = boxes_t[keep], labels_t[keep], scores_t[keep]
    if len(boxes_t):
        keep_idx = torch_nms(boxes_t, scores_t, nms_thr)[:max_det]
        boxes_t, labels_t, scores_t = boxes_t[keep_idx], labels_t[keep_idx], scores_t[keep_idx]

    for box, label_id, score in zip(boxes_t, labels_t, scores_t):
        conf     = float(score)
        cls_id   = int(label_id)
        label    = labels_map.get(cls_id, str(cls_id))

        # Bỏ qua background (id=0) và confidence thấp
        if label.lower() == "background":
            continue
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(orig_w - 1, round(x1 * scale_x))))
        x2 = int(max(0, min(orig_w - 1, round(x2 * scale_x))))
        y1 = int(max(0, min(orig_h - 1, round(y1 * scale_y))))
        y2 = int(max(0, min(orig_h - 1, round(y2 * scale_y))))
        if x2 <= x1 or y2 <= y1:
            continue

        lv    = classify_label(label)
        level = max(level, lv)
        if conf > max_conf:
            max_conf = conf

        color = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)
        if lv == 2:
            color = (0, 0, 255)
        elif lv == 1:
            color = (0, 165, 255)

        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        text  = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(ann, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(ann, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        det.append({
            "label": label,
            "conf":  round(conf, 3),
            "bbox":  [x1, y1, x2, y2],
            "level": lv,
            "algo":  "faster_rcnn",
        })

    return ann, det, level, (max_conf if max_conf > 0 else None)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Non-Maximum Suppression đơn giản"""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if _iou(best.bbox, d.bbox) < iou_threshold or d.label != best.label
        ]

    return kept


def _iou(box1: Tuple, box2: Tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def get_benchmark_results(algorithm: Algorithm) -> dict:
    """Trả về kết quả benchmark cho thuật toán"""
    return ALGORITHM_CONFIG[algorithm]


def compare_algorithms() -> dict:
    """So sánh tất cả thuật toán, xếp hạng theo rank"""
    return {
        algo.value: ALGORITHM_CONFIG[algo]
        for algo in sorted(Algorithm, key=lambda a: ALGORITHM_CONFIG[a]["rank"])
    }


def simulate_detection(
    frame:          np.ndarray,
    algorithm:      Algorithm = Algorithm.SSD,
    conf_threshold: float = 0.5,
    iou_threshold:  float = 0.45,
) -> FrameResult:
    """Mô phỏng nhận diện — dùng để test khi chưa có model thật"""
    cfg = ALGORITHM_CONFIG[algorithm]
    t0  = time.time()

    latency = max(0, np.random.normal(cfg["latency_mean"], cfg["latency_std"])) / 1000
    time.sleep(latency * 0.01)

    detections: List[Detection] = []
    h, w = frame.shape[:2] if frame.ndim == 3 else (480, 640)

    for _ in range(random.randint(1, 5)):
        label    = random.choice(CLASSES)
        base_conf = cfg.get("precision") if label == "accident" else cfg.get("map50")
        if base_conf is None:
            base_conf = 0.7
        conf     = float(np.clip(np.random.normal(base_conf, 0.05), 0.3, 0.99))
        if conf < conf_threshold:
            continue

        x1 = random.randint(0, w - 100)
        y1 = random.randint(0, h - 80)
        x2 = min(x1 + random.randint(60, 200), w)
        y2 = min(y1 + random.randint(50, 150), h)

        detections.append(Detection(
            bbox=(x1, y1, x2, y2),
            label=label,
            confidence=conf,
            algorithm=algorithm.value,
        ))

    elapsed = (time.time() - t0) * 1000
    return FrameResult(
        detections=detections,
        fps=min(1000 / max(elapsed, 1), (cfg.get("fps_mean") or 30.0) * 1.2),
        latency_ms=elapsed,
        algorithm=algorithm.value,
        frame_id=random.randint(0, 999999),
    )
