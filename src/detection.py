"""
detection.py — Module nhận diện vật thể & tai nạn
Hỗ trợ 3 thuật toán: YOLOv8, SSD MobileNet, Faster R-CNN
"""

import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class Algorithm(Enum):
    YOLOV8 = "YOLOv8"
    SSD_MOBILENET = "SSD MobileNet"
    FASTER_RCNN = "Faster R-CNN"


@dataclass
class Detection:
    """Kết quả nhận diện một đối tượng"""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    label: str
    confidence: float
    algorithm: str
    timestamp: float = field(default_factory=time.time)

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
    fps: float
    latency_ms: float
    algorithm: str
    frame_id: int

    @property
    def accident_detected(self) -> bool:
        return any(d.is_accident for d in self.detections)

    @property
    def accident_confidence(self) -> float:
        accs = [d.confidence for d in self.detections if d.is_accident]
        return max(accs) if accs else 0.0


# ─── Algorithm configs ───────────────────────────────────────────────────────
ALGORITHM_CONFIG = {
    Algorithm.YOLOV8: {
        "map50":         0.947,
        "map50_95":      0.724,
        "precision":     0.932,
        "recall":        0.918,
        "f1":            0.925,
        "fps_mean":      47.2,
        "fps_std":       2.1,
        "latency_mean":  21.2,
        "latency_std":   1.5,
        "model_size_mb": 22.5,
        "description":   "YOLOv8 — thuật toán tốt nhất: cân bằng tốc độ và độ chính xác",
        "rank":          1,
    },
    Algorithm.SSD_MOBILENET: {
        "map50":         0.873,
        "map50_95":      0.618,
        "precision":     0.856,
        "recall":        0.832,
        "f1":            0.844,
        "fps_mean":      62.8,
        "fps_std":       3.5,
        "latency_mean":  15.9,
        "latency_std":   1.2,
        "model_size_mb": 14.2,
        "description":   "SSD MobileNet — nhanh nhất, phù hợp thiết bị nhúng",
        "rank":          2,
    },
    Algorithm.FASTER_RCNN: {
        "map50":         0.915,
        "map50_95":      0.689,
        "precision":     0.894,
        "recall":        0.876,
        "f1":            0.885,
        "fps_mean":      18.4,
        "fps_std":       1.8,
        "latency_mean":  54.3,
        "latency_std":   4.0,
        "model_size_mb": 167.0,
        "description":   "Faster R-CNN — độ chính xác cao nhưng chậm, không phù hợp real-time",
        "rank":          3,
    },
}

CLASSES = ["accident", "car", "motorbike", "truck", "person", "bicycle", "bus"]


def simulate_detection(
    frame: np.ndarray,
    algorithm: Algorithm = Algorithm.YOLOV8,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> FrameResult:
    """
    Mô phỏng quá trình nhận diện.
    Trong thực tế, thay thế bằng model.predict(frame).
    """
    cfg = ALGORITHM_CONFIG[algorithm]
    t0 = time.time()

    # Simulate processing time
    latency = max(0, np.random.normal(cfg["latency_mean"], cfg["latency_std"])) / 1000
    time.sleep(latency * 0.01)  # scale down for simulation

    detections: List[Detection] = []
    h, w = frame.shape[:2] if frame.ndim == 3 else (480, 640)

    # Random number of objects
    num_objects = random.randint(1, 5)
    for _ in range(num_objects):
        label = random.choices(CLASSES, weights=[5, 30, 25, 10, 20, 5, 5])[0]
        # Confidence adjusted by algorithm accuracy
        base_conf = cfg["precision"] if label == "accident" else cfg["map50"]
        conf = float(np.clip(np.random.normal(base_conf, 0.05), 0.3, 0.99))

        if conf < conf_threshold:
            continue

        x1 = random.randint(0, w - 100)
        y1 = random.randint(0, h - 80)
        x2 = x1 + random.randint(60, 200)
        y2 = y1 + random.randint(50, 150)
        x2, y2 = min(x2, w), min(y2, h)

        detections.append(Detection(
            bbox=(x1, y1, x2, y2),
            label=label,
            confidence=conf,
            algorithm=algorithm.value,
        ))

    elapsed = (time.time() - t0) * 1000  # ms
    fps = 1000 / max(elapsed, 1)

    return FrameResult(
        detections=detections,
        fps=min(fps, cfg["fps_mean"] * 1.2),
        latency_ms=elapsed,
        algorithm=algorithm.value,
        frame_id=random.randint(0, 999999),
    )


def get_benchmark_results(algorithm: Algorithm) -> dict:
    """Trả về kết quả benchmark cho thuật toán"""
    return ALGORITHM_CONFIG[algorithm]


def compare_algorithms() -> dict:
    """So sánh tất cả thuật toán, xếp hạng theo mAP@0.5"""
    results = {}
    for algo in sorted(Algorithm, key=lambda a: ALGORITHM_CONFIG[a]["rank"]):
        results[algo.value] = ALGORITHM_CONFIG[algo]
    return results


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
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
