"""
utils.py — Các hàm hỗ trợ xử lý hình ảnh
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


def letterbox(img: np.ndarray, new_shape: Tuple[int,int] = (640, 640),
              color: Tuple = (114,114,114)) -> np.ndarray:
    """Resize + pad ảnh giữ tỉ lệ khung hình (letterbox)"""
    shape = img.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img


def preprocess(frame: np.ndarray, size: int = 640) -> np.ndarray:
    """Chuẩn bị frame đầu vào cho mô hình"""
    img = letterbox(frame, (size, size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis]  # add batch dim


def draw_detections(frame: np.ndarray, detections: list,
                    color_map: Optional[dict] = None) -> np.ndarray:
    """Vẽ bounding box và nhãn lên frame"""
    default_colors = {
        "accident":  (0, 0, 255),    # đỏ
        "car":       (0, 255, 128),  # xanh lá
        "motorbike": (255, 165, 0),  # cam
        "person":    (0, 200, 255),  # xanh dương
        "truck":     (200, 0, 255),  # tím
    }
    colors = color_map or default_colors

    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label, conf = det.label, det.confidence
        color = colors.get(label.lower(), (200, 200, 200))

        # Draw box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(out, text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

    return out


def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Vẽ quỹ đạo chuyển động của các đối tượng"""
    out = frame.copy()
    for track in tracks:
        hist = list(track.history)
        for i in range(1, len(hist)):
            pt1 = (int(hist[i-1][0]), int(hist[i-1][1]))
            pt2 = (int(hist[i][0]),   int(hist[i][1]))
            alpha = i / len(hist)
            color = (int(255*alpha), int(100*alpha), 255)
            cv2.line(out, pt1, pt2, color, 2)
        # Draw track ID
        cx, cy = int(track.center[0]), int(track.center[1])
        cv2.putText(out, f"ID:{track.track_id}", (cx-15, cy-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    return out


def add_hud(frame: np.ndarray, fps: float, algorithm: str,
            accident: bool = False, n_objects: int = 0) -> np.ndarray:
    """Thêm HUD (Heads-Up Display) lên góc trên trái frame"""
    out = frame.copy()
    color = (0, 0, 255) if accident else (0, 255, 128)

    # Semi-transparent overlay
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (280, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    cv2.putText(out, f"Algorithm: {algorithm}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(out, f"FPS: {fps:.1f}", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(out, f"Objects: {n_objects}", (8, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    status = "ACCIDENT DETECTED" if accident else "NORMAL"
    cv2.putText(out, status, (8, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return out


def compute_iou(b1: Tuple, b2: Tuple) -> float:
    """Tính IoU giữa 2 bounding box"""
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def resize_maintain_aspect(img: np.ndarray,
                            target_w: int = 1280) -> np.ndarray:
    """Resize giữ tỉ lệ theo chiều rộng"""
    h, w = img.shape[:2]
    target_h = int(h * target_w / w)
    return cv2.resize(img, (target_w, target_h))
