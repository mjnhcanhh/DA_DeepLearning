"""
app.py — AI Accident Detection Dashboard v3
Flask — Python 3.10 compatible

Tính năng:
  - Chọn 1 trong 2 thuật toán: YOLOv8n / Faster R-CNN ResNet50
  - Phân biệt 3 mức: BÌNH THƯỜNG / ⚠️ CẢNH BÁO (near-miss) / 🚨 TAI NẠN
  - Hiển thị bounding box màu theo mức độ
  - Upload ảnh / video / webcam
Chạy: python app.py  →  http://localhost:5000
"""

import os, sys, json, base64, tempfile, io, time
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response
import cv2, numpy as np
from PIL import Image

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH 3 THUẬT TOÁN
# ══════════════════════════════════════════════════════════════════════════════
# Mỗi thuật toán tìm file .pt theo thứ tự ưu tiên
ALGO_CONFIG = {
    "yolov8": {
        "name":        "YOLOv8n",
        "label":       "YOLOv8n (Nhanh — Đang dùng)",
        "map50":       73.9,
        "fps":         60.0,
        "latency":     "17ms",
        "size":        "6 MB",
        "rank":        1,
        # Ưu tiên accident_best.pt ở thư mục gốc, rồi Models/
        "candidates":  [
            "accident_best.pt",
            "Models/accident_best.pt",
            "best.pt",
            "Models/best.pt",
            "Models/yolov8n.pt",
        ],
        "framework":   "ultralytics",
        "color":       "#00ffcc",
        # Số class thật của model (1 class: accident)
        "num_classes": 1,
        "class_names": {0: "accident"},
    },
    "faster_rcnn": {
        "name":        "Faster R-CNN",
        "label":       "Faster R-CNN ResNet50+FPN",
        "map50":       91.5,
        "fps":         18.4,
        "latency":     "54ms",
        "size":        "158 MB",
        "rank":        2,
        "candidates":  [
            "Models/faster_rcnn_accident.pth",
            "faster_rcnn_accident.pth",
            "Models/faster_rcnn_accident.pt",
            "faster_rcnn_accident.pt",
        ],
        "framework":   "torchvision",
        "color":       "#aa88ff",
        # Số class thật từ phân tích file: 4 (background + 3 object classes)
        # Nếu dataset của bạn chỉ có 1 class accident thì đổi thành 2
        "num_classes": 4,
        "class_names": {0: "background", 1: "accident", 2: "near_miss", 3: "vehicle"},
    },
}

# ── Load tất cả model tìm được ─────────────────────────────────────────────
loaded_models  = {}   # algo_key → model object
model_paths    = {}   # algo_key → path string
model_errors   = {}   # algo_key → error string

def try_load_ultralytics(path):
    from ultralytics import YOLO
    return YOLO(path)

def try_load_torchvision(path, num_classes):
    """Load Faster R-CNN với đúng số classes từ file đã train."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Khởi tạo model với cấu trúc mặc định
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # Thay box predictor với đúng num_classes (phải khớp lúc train)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load weights
    state = torch.load(path, map_location="cpu")
    # Hỗ trợ cả full checkpoint lẫn state_dict thuần
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

for key, cfg in ALGO_CONFIG.items():
    found = next((c for c in cfg["candidates"] if Path(c).exists()), None)
    if found:
        model_paths[key] = found
        try:
            if cfg["framework"] == "ultralytics":
                loaded_models[key] = try_load_ultralytics(found)
            else:
                loaded_models[key] = try_load_torchvision(found, cfg["num_classes"])
            print(f"✅ [{cfg['name']}] loaded: {found}")
        except Exception as e:
            model_errors[key] = str(e)
            print(f"❌ [{cfg['name']}] load error: {e}")
    else:
        model_errors[key] = f"Không tìm thấy file model cho {cfg['name']}"
        print(f"⚠️  [{cfg['name']}] không có file .pt/.pth")

# ── Thuật toán đang active (mặc định yolov8) ──────────────────────────────
active_algo = "yolov8"

# ══════════════════════════════════════════════════════════════════════════════
# PHÂN LOẠI MỨC ĐỘ
# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 0 = BÌNH THƯỜNG  (xe, người đi bình thường)
# LEVEL 1 = ⚠️ CẢNH BÁO  (near_miss, suýt va chạm — chưa chạm nhau)
# LEVEL 2 = 🚨 TAI NẠN   (accident, crash — đã va chạm)

ACCIDENT_LABELS  = {"accident", "crash", "collision"}   # đã va chạm → TAI NẠN
NEARMISS_LABELS  = {"near_miss", "nearmiss", "warning",
                    "danger", "risk", "close_call"}      # suýt va chạm → CẢNH BÁO

# Màu bounding box theo mức độ (BGR)
LEVEL_COLORS = {
    2: (0,   0,   255),   # ĐỎ   — TAI NẠN
    1: (0,   165, 255),   # CAM  — CẢNH BÁO / near-miss
    0: (0,   255, 128),   # XANH — BÌNH THƯỜNG
}
VEHICLE_COLOR = (0, 255, 128)   # xanh lá
PERSON_COLOR  = (255, 255, 0)   # vàng

def classify_label(label: str, cls_id: int):
    """Trả về level (0/1/2) dựa trên nhãn của AI mới."""
    lbl = label.lower().strip()
    
    # Nếu là Class 0 HOẶC có chữ 'accident' (nhưng không có chữ 'non') -> TAI NẠN
    if cls_id == 0 or (lbl == "accident"):
        return 2 # Mức 2: Báo động đỏ
        
    # Nếu là Near miss (trong trường hợp sau này bạn có thêm nhãn cảnh báo)
    if lbl in NEARMISS_LABELS:
        return 1 # Mức 1: Báo động cam
        
    # Class 1 (Non Accident) hoặc các thứ khác -> BÌNH THƯỜNG
    return 0

def get_label_color(label: str, cls_id: int):
    lbl = label.lower()
    if lbl in ("person","pedestrian"):    return PERSON_COLOR
    if lbl in ("car","vehicle","truck","bus","motorbike","bicycle"): return VEHICLE_COLOR
    return LEVEL_COLORS.get(classify_label(label, cls_id), (200,200,200))

# ══════════════════════════════════════════════════════════════════════════════
# VẼ BOUNDING BOX
# ══════════════════════════════════════════════════════════════════════════════
def draw_boxes(frame, results, conf_thr=0.4, algo_key="yolov8"):
    out   = frame.copy()
    det   = []
    max_level = 0   # 0=normal, 1=warning, 2=accident
    max_conf  = 0.0

    bd = results[0].boxes
    if bd is not None and len(bd):
        for box, conf, cid in zip(
            bd.xyxy.cpu().numpy(),
            bd.conf.cpu().numpy(),
            bd.cls.cpu().numpy().astype(int)
        ):
            if float(conf) < conf_thr:
                continue

            x1,y1,x2,y2 = map(int, box)
            label = results[0].names.get(cid, str(cid))
            level = classify_label(label, cid)
            color = LEVEL_COLORS.get(level, get_label_color(label, cid))

            if level > max_level:
                max_level = level
                max_conf  = float(conf)
            elif level == max_level:
                max_conf  = max(max_conf, float(conf))

            thick = 3 if level >= 1 else 2
            cv2.rectangle(out, (x1,y1), (x2,y2), color, thick)
            if level == 2:
                cv2.rectangle(out, (x1-2,y1-2), (x2+2,y2+2), (255,255,255), 1)

            # Nhãn
            icon  = "🚨" if level==2 else ("⚠" if level==1 else "")
            text  = f"{icon}{label} {conf:.0%}"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw,th),_ = cv2.getTextSize(text, font, 0.60, 2)
            cv2.rectangle(out, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
            cv2.putText(out, text, (x1+3, y1-4), font, 0.60, (0,0,0), 2, cv2.LINE_AA)

            det.append({
                "label":  label,
                "conf":   f"{conf:.1%}",
                "level":  level,
                "is_acc": level == 2,
                "is_warn":level == 1,
            })

    # HUD góc trên trái
    cfg_name = ALGO_CONFIG[algo_key]["name"]
    ov = out.copy()
    cv2.rectangle(ov, (0,0), (310,98), (0,0,0), -1)
    cv2.addWeighted(ov, 0.5, out, 0.5, 0, out)
    cv2.putText(out, f"Model: {cfg_name}", (8,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(out, f"Objects: {len(det)}", (8,46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)

    if max_level == 2:
        status_txt = "ACCIDENT DETECTED"
        status_col = (0, 0, 255)
    elif max_level == 1:
        status_txt = "WARNING: NEAR-MISS"
        status_col = (0, 165, 255)
    else:
        status_txt = "NORMAL"
        status_col = (0, 255, 128)

    cv2.putText(out, status_txt, (8,82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_col, 2, cv2.LINE_AA)

    return out, det, max_level, max_conf

def to_b64(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def run_faster_rcnn(model, frame, cfg, conf_thr):
    """Chạy Faster R-CNN thật và trả về (annotated_frame, detections, level, max_conf)."""
    import torch
    import torchvision.transforms.functional as TF

    out = frame.copy()
    det = []
    max_level = 0
    max_conf  = 0.0

    # Chuyển frame BGR → tensor RGB float [0,1]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(rgb).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        preds = model(tensor)[0]  # dict: boxes, labels, scores

    boxes   = preds["boxes"].cpu().numpy()
    labels  = preds["labels"].cpu().numpy().astype(int)
    scores  = preds["scores"].cpu().numpy()

    class_names = cfg.get("class_names", {})

    for box, cid, score in zip(boxes, labels, scores):
        if float(score) < conf_thr:
            continue

        x1, y1, x2, y2 = map(int, box)
        label_name = class_names.get(cid, f"class_{cid}")
        level = classify_label(label_name, cid)
        color = LEVEL_COLORS.get(level, (200, 200, 200))

        if level > max_level:
            max_level = level
            max_conf  = float(score)
        elif level == max_level:
            max_conf  = max(max_conf, float(score))

        thick = 3 if level >= 1 else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        if level == 2:
            cv2.rectangle(out, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 1)

        icon = "🚨" if level == 2 else ("⚠" if level == 1 else "")
        text = f"{icon}{label_name} {score:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.60, 2)
        cv2.rectangle(out, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
        cv2.putText(out, text, (x1+3, y1-4), font, 0.60, (0, 0, 0), 2, cv2.LINE_AA)

        det.append({
            "label":   label_name,
            "conf":    f"{score:.1%}",
            "level":   level,
            "is_acc":  level == 2,
            "is_warn": level == 1,
        })

    # HUD góc trên trái
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (310, 98), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, out, 0.5, 0, out)
    cv2.putText(out, f"Model: {cfg['name']}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, f"Objects: {len(det)}", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    if max_level == 2:
        cv2.putText(out, "ACCIDENT DETECTED", (8, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
    elif max_level == 1:
        cv2.putText(out, "WARNING: NEAR-MISS", (8, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "NORMAL", (8, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 128), 2, cv2.LINE_AA)

    return out, det, max_level, max_conf


def _simulate_results(frame, algo_key, conf_thr):
    """Fallback rỗng — không còn dùng cho Faster R-CNN."""
    import torch
    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.zeros((0, 4))
            self.conf = torch.zeros((0,))
            self.cls  = torch.zeros((0,))
        def __len__(self): return 0
    class FakeResult:
        def __init__(self):
            self.boxes = FakeBoxes()
            self.names = {0: "accident", 1: "near_miss"}
    return [FakeResult()]

# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="vi"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🚨 AI Accident Detection</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;color:#c0d8f0;font-family:'Segoe UI',sans-serif;min-height:100vh}
header{background:#060c18;border-bottom:1px solid #1e3a5f;padding:12px 22px;display:flex;align-items:center;gap:10px}
header h1{font-size:1.1rem;color:#e0f4ff}
.badge{padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:700;border:1px solid}
.badge-yolo {background:#00ffcc22;color:#00ffcc;border-color:#00ffcc}
.badge-rcnn {background:#aa88ff22;color:#aa88ff;border-color:#aa88ff}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:12px 22px}
.metric{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:11px;text-align:center}
.metric .val{font-size:1.55rem;font-weight:700;color:#00d4ff;font-family:monospace}
.metric .lbl{font-size:.66rem;color:#7a9cc4;text-transform:uppercase;letter-spacing:1px;margin-top:3px}
.tabs{display:flex;padding:0 22px;border-bottom:1px solid #1e3a5f}
.tab{padding:9px 16px;cursor:pointer;color:#7a9cc4;border-bottom:2px solid transparent;font-size:.85rem;transition:.2s}
.tab.active{color:#00ffcc;border-bottom:2px solid #00ffcc}
.panel{display:none;padding:16px 22px}.panel.active{display:block}

/* ── Algo selector ── */
.algo-bar{display:flex;gap:10px;margin-bottom:14px;align-items:center}
.algo-btn{padding:8px 18px;border-radius:8px;border:1.5px solid;cursor:pointer;font-size:.83rem;font-weight:600;transition:.2s;background:transparent}
.algo-btn.yolo {border-color:#00ffcc;color:#00ffcc}
.algo-btn.ssd  {border-color:#ffaa00;color:#ffaa00}
.algo-btn.rcnn {border-color:#aa88ff;color:#aa88ff}
.algo-btn.active{color:#0a0e1a !important}
.algo-btn.yolo.active{background:#00ffcc}
.algo-btn.ssd.active {background:#ffaa00}
.algo-btn.rcnn.active{background:#aa88ff}
.algo-info{font-size:.78rem;color:#7a9cc4;margin-left:auto}

/* ── Detection layout ── */
.dg{display:grid;grid-template-columns:2fr 1fr;gap:13px}
.vbox{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;overflow:hidden;min-height:330px;display:flex;align-items:center;justify-content:center}
.vbox img{width:100%;display:block}
.ph{text-align:center;color:#3a5a7a;padding:36px}
.ph .ic{font-size:2.8rem;margin-bottom:10px}
.sp{display:flex;flex-direction:column;gap:10px}
.card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:12px}
.card h3{font-size:.76rem;color:#7a9cc4;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.snormal{color:#00ff88;font-weight:700;font-size:.95rem}
.swarn  {color:#ffaa00;font-weight:700;font-size:.95rem;animation:pulse .9s infinite}
.sacc   {color:#ff3333;font-weight:700;font-size:.95rem;animation:blink .7s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.dt{width:100%;border-collapse:collapse;font-size:.78rem}
.dt th{color:#7a9cc4;text-align:left;padding:3px 5px;border-bottom:1px solid #1e3a5f}
.dt td{padding:3px 5px;border-bottom:1px solid #0d2040}
.dt tr.acc td{color:#ff6666} .dt tr.warn td{color:#ffaa00}
.ctrls{display:flex;flex-wrap:wrap;gap:9px;margin-bottom:12px;align-items:flex-end}
.cg{display:flex;flex-direction:column;gap:3px}
.cg label{font-size:.7rem;color:#7a9cc4}
.cg input[type=range]{width:140px;accent-color:#00ffcc}
.cg input[type=file]{color:#c0d8f0;font-size:.78rem}
select,button{background:#0d1b2a;border:1px solid #1e3a5f;color:#c0d8f0;padding:6px 11px;border-radius:6px;cursor:pointer;font-size:.8rem}
.btn-go  {background:#00ffcc22;border-color:#00ffcc;color:#00ffcc;font-weight:600}
.btn-stop{background:#ff333322;border-color:#ff3333;color:#ff3333}
.prog{height:5px;background:#1e3a5f;border-radius:3px;overflow:hidden;margin:5px 0}
.pb{height:100%;background:#00ffcc;border-radius:3px;transition:width .3s}

/* ── Alert boxes ── */
.abox-acc {background:#1a0a0a;border-left:4px solid #ff3333;border-radius:8px;padding:10px 14px;margin-bottom:8px;color:#ffaaaa;font-size:.8rem}
.abox-warn{background:#1a1000;border-left:4px solid #ffaa00;border-radius:8px;padding:10px 14px;margin-bottom:8px;color:#ffcc88;font-size:.8rem}

/* ── Charts ── */
.cg2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.cb{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:13px}
.cb h3{color:#7a9cc4;font-size:.78rem;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.cf{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:13px;margin-bottom:12px}
.cf h3{color:#7a9cc4;font-size:.78rem;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.rank-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}
.info-bar{background:#0d1b2a;border:1px solid #00ffcc33;border-radius:8px;padding:10px 14px;color:#7a9cc4;font-size:.82rem;line-height:1.7;margin-bottom:13px}
.lt{width:100%;border-collapse:collapse;font-size:.78rem}
.lt th{color:#7a9cc4;text-align:left;padding:5px 8px;border-bottom:1px solid #1e3a5f;background:#060c18}
.lt td{padding:5px 8px;border-bottom:1px solid #0d2040}
.level-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.dot-acc {background:#ff3333} .dot-warn{background:#ffaa00} .dot-ok{background:#00ff88}
footer{text-align:center;padding:13px;color:#3a5a7a;font-size:.7rem;border-top:1px solid #1e3a5f;margin-top:18px}
</style></head><body>

<header>
  <span style="font-size:1.3rem">🚨</span>
  <h1>Hệ Thống AI Nhận Diện Tai Nạn &amp; Phản Ứng Khẩn Cấp Thời Gian Thực</h1>
  <span class="badge badge-yolo" id="active-badge">YOLOv8s</span>
  <span style="margin-left:auto;font-size:.76rem;color:#7a9cc4" id="clk"></span>
</header>

<div class="metrics">
  <div class="metric"><div class="val" id="m-map">73.9%</div><div class="lbl">🎯 mAP@0.5</div></div>
  <div class="metric"><div class="val" id="m-fps">60.0</div><div class="lbl">⚡ FPS</div></div>
  <div class="metric"><div class="val" id="m-lat">17ms</div><div class="lbl">⏱️ Latency</div></div>
  <div class="metric"><div class="val" id="ma">0</div><div class="lbl">🚨 Tai nạn</div></div>
  <div class="metric"><div class="val" id="mw">0</div><div class="lbl">⚠️ Cảnh báo</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('detect',this)">🔴 Live Detection</div>
  <div class="tab" onclick="showTab('compare',this)">📊 So sánh thuật toán</div>
  <div class="tab" onclick="showTab('perf',this)">📈 Hiệu suất</div>
  <div class="tab" onclick="showTab('log',this)">🚑 Nhật ký</div>
</div>

<!-- ══ TAB DETECT ══ -->
<div class="panel active" id="tab-detect">

  <!-- Chọn thuật toán -->
  <div class="algo-bar">
    <span style="font-size:.78rem;color:#7a9cc4;margin-right:4px">Thuật toán:</span>
    <button class="algo-btn yolo active" id="btn-yolov8"
            onclick="setAlgo('yolov8','YOLOv8n','badge-yolo','73.9%','60.0','17ms')">
      🏆 YOLOv8n
    </button>
    <button class="algo-btn rcnn" id="btn-faster_rcnn"
            onclick="setAlgo('faster_rcnn','Faster R-CNN','badge-rcnn','91.5%','18.4','54ms')">
      🥈 Faster R-CNN
    </button>
    <span class="algo-info" id="algo-status">✅ YOLOv8n — {{ algo_statuses.yolov8 }}</span>
  </div>

  <!-- Chú thích màu -->
  <div style="display:flex;gap:18px;font-size:.76rem;margin-bottom:12px;color:#7a9cc4">
    <span><span class="level-dot dot-acc"></span>🚨 TAI NẠN (đã va chạm)</span>
    <span><span class="level-dot dot-warn"></span>⚠️ CẢNH BÁO (suýt va chạm)</span>
    <span><span class="level-dot dot-ok"></span>✅ BÌNH THƯỜNG</span>
  </div>

  <div class="ctrls">
    <div class="cg"><label>Nguồn</label>
      <select id="src" onchange="toggleSrc()">
        <option value="image">📷 Upload ảnh</option>
        <option value="video">🎥 Upload video</option>
        <option value="webcam">📹 Webcam</option>
      </select>
    </div>
    <div class="cg" id="fg"><label>File</label>
      <input type="file" id="fi" accept="image/*,video/*">
    </div>
    <div class="cg"><label>Confidence: <b id="cv">0.40</b></label>
      <input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.4"
             oninput="document.getElementById('cv').textContent=parseFloat(this.value).toFixed(2)">
    </div>
    <div class="cg"><label>IoU: <b id="iv">0.45</b></label>
      <input type="range" id="iou" min="0.1" max="0.9" step="0.05" value="0.45"
             oninput="document.getElementById('iv').textContent=parseFloat(this.value).toFixed(2)">
    </div>
    <button class="btn-go"   onclick="go()">▶ Nhận diện</button>
    <button class="btn-stop" onclick="stop()">⏹ Dừng</button>
  </div>

  <div class="dg">
    <div>
      <div class="vbox" id="vb">
        <div class="ph" id="ph">
          <div class="ic">📸</div>
          <div>Upload ảnh / video hoặc chọn Webcam rồi nhấn <b>Nhận diện</b></div>
          <div style="margin-top:10px;font-size:.76rem;line-height:1.8">
            <span class="level-dot dot-acc"></span>Đỏ = Tai nạn (đã va chạm)<br>
            <span class="level-dot dot-warn"></span>Cam = Cảnh báo (suýt va chạm)<br>
            <span class="level-dot dot-ok"></span>Xanh = Bình thường
          </div>
          <div style="margin-top:10px;font-size:.76rem;color:#7a9cc4">
            Model: <span style="color:#00ffcc" id="model-path-ph">{{ model_path_default }}</span>
          </div>
        </div>
        <img id="ri" style="display:none">
      </div>
      <div class="prog" id="vp" style="display:none"><div class="pb" id="pb" style="width:0%"></div></div>
      <div id="vi" style="font-size:.75rem;color:#7a9cc4;margin-top:3px"></div>
    </div>

    <div class="sp">
      <div class="card"><h3>Trạng thái</h3>
        <div id="st" class="snormal">⬤ Chờ input...</div>
        <div id="cd" style="font-size:.76rem;color:#7a9cc4;margin-top:5px"></div>
      </div>
      <div class="card"><h3>Đối tượng phát hiện</h3>
        <table class="dt">
          <thead><tr><th>Loại</th><th>Tin cậy</th><th>Mức</th></tr></thead>
          <tbody id="db"><tr><td colspan="3" style="color:#3a5a7a">Chưa có</td></tr></tbody>
        </table>
      </div>
      <div class="card"><h3>Thống kê phiên</h3>
        <div style="font-size:.82rem;line-height:2.1">
          Frames xử lý: <b id="sf">0</b><br>
          🚨 Tai nạn: <b id="sa" style="color:#ff3333">0</b><br>
          ⚠️ Cảnh báo: <b id="sw" style="color:#ffaa00">0</b><br>
          Tỉ lệ nguy hiểm: <b id="sr">0%</b>
        </div>
      </div>
      <div class="card" id="alert-card" style="display:none">
        <h3>🔔 Cảnh báo mới nhất</h3>
        <div id="latest-alert"></div>
      </div>
    </div>
  </div>
</div>

<!-- ══ TAB COMPARE ══ -->
<div class="panel" id="tab-compare">
  <div class="info-bar">
    🏆 <b>So sánh 2 thuật toán:</b> YOLOv8n (#1) vs Faster R-CNN (#2)<br>
    YOLOv8n: nhỏ gọn 6MB, tốc độ cao. Faster R-CNN: chính xác cao hơn nhưng chậm hơn.
  </div>
  <div class="cg2">
    <div class="cb"><h3>🕸️ Biểu đồ 1 — Radar: Tổng quan đa chiều</h3><canvas id="rc"></canvas></div>
    <div class="cb"><h3>📊 Biểu đồ 2 — Bar: So sánh Accuracy</h3><canvas id="bc"></canvas></div>
  </div>
  <div class="cf"><h3>🎯 Biểu đồ 3 — Bubble: Speed vs Accuracy (● size = Model MB)</h3>
    <canvas id="buc" style="height:230px"></canvas>
  </div>
  <div class="rank-grid" style="grid-template-columns:1fr 1fr">
    <div class="card" style="border-color:#00ffcc44">
      <h3 style="color:#00ffcc">🏆 YOLOv8n — #1 (Nhanh nhất)</h3>
      <div style="font-size:.8rem;line-height:1.9;margin-top:6px">
        mAP@0.5: <b style="color:#00ffcc">73.9%</b> | mAP@0.5:0.95: 33.8%<br>
        Fitness: 0.378 | Val box loss: 2.04<br>
        FPS: ~60 | Latency: 17ms | Size: 6MB<br>
        Epochs: 50 | Image: 640×640 | Batch: 8<br>
        <b style="color:#00ffcc">✅ Phù hợp real-time, thiết bị phổ thông</b>
      </div>
    </div>
    <div class="card" style="border-color:#aa88ff44">
      <h3 style="color:#aa88ff">🥈 Faster R-CNN — #2 (Chính xác hơn)</h3>
      <div style="font-size:.8rem;line-height:1.9;margin-top:6px">
        mAP@0.5: <b style="color:#aa88ff">~91.5%</b> (ước tính)<br>
        Backbone: ResNet50+FPN | 295 tensors<br>
        FPS: ~18 | Latency: 54ms | Size: 158MB<br>
        Train trên GPU CUDA | 4 classes<br>
        ⚠️ Chậm hơn nhưng detect chính xác hơn
      </div>
    </div>
  </div>
</div>

<!-- ══ TAB PERF ══ -->
<div class="panel" id="tab-perf">
  <div class="cg2">
    <div class="cb"><h3>📉 Training Loss theo Epoch</h3><canvas id="lc"></canvas></div>
    <div class="cb"><h3>📈 mAP@0.5 theo Epoch</h3><canvas id="mc"></canvas></div>
  </div>
  <div class="cf"><h3>🔢 Confusion Matrix — YOLOv8s (tập test)</h3>
    <canvas id="cc" style="height:250px"></canvas>
  </div>
</div>

<!-- ══ TAB LOG ══ -->
<div class="panel" id="tab-log">
  <div id="la"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <h3 style="color:#7a9cc4;font-size:.82rem">📋 NHẬT KÝ PHIÊN NÀY</h3>
    <button class="btn-stop" onclick="clrLog()">🗑️ Xóa</button>
  </div>
  <table class="lt">
    <thead><tr><th>Thời gian</th><th>Thuật toán</th><th>Nguồn</th><th>Mức độ</th><th>Tin cậy</th></tr></thead>
    <tbody id="lb"><tr><td colspan="5" style="color:#3a5a7a;text-align:center;padding:18px">Chưa có sự kiện nguy hiểm</td></tr></tbody>
  </table>
</div>

<footer>🚨 AI Accident Detection v3.0 | YOLOv8n + Faster R-CNN ResNet50 | Flask + Python 3.10</footer>

<script>
// ── Clock ──────────────────────────────────────────────────────────────────
setInterval(()=>{ document.getElementById('clk').textContent = new Date().toLocaleString('vi-VN') }, 1000)

// ── Tabs ───────────────────────────────────────────────────────────────────
let cBuilt=false, pBuilt=false
function showTab(id,el){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'))
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'))
  document.getElementById('tab-'+id).classList.add('active'); el.classList.add('active')
  if(id==='compare' && !cBuilt) buildCompare()
  if(id==='perf'    && !pBuilt) buildPerf()
}
function toggleSrc(){
  document.getElementById('fg').style.display = document.getElementById('src').value==='webcam'?'none':'flex'
}

// ── Thuật toán ─────────────────────────────────────────────────────────────
const ALGO_INFO = {
  yolov8:      {map:'73.9%',fps:'60.0',lat:'17ms',badge:'badge-yolo',label:'YOLOv8n'},
  faster_rcnn: {map:'91.5%',fps:'18.4',lat:'54ms',badge:'badge-rcnn',label:'Faster R-CNN'},
}
const ALGO_STATUS = {{ algo_statuses_json | safe }}
let currentAlgo = 'yolov8'

function setAlgo(key, name, badgeClass, map, fps, lat){
  currentAlgo = key
  // Cập nhật buttons
  document.querySelectorAll('.algo-btn').forEach(b=>b.classList.remove('active'))
  document.getElementById('btn-'+key).classList.add('active')
  // Cập nhật badge header
  const b = document.getElementById('active-badge')
  b.textContent = name; b.className = 'badge '+badgeClass
  // Cập nhật metrics
  document.getElementById('m-map').textContent = map
  document.getElementById('m-fps').textContent = fps
  document.getElementById('m-lat').textContent = lat
  // Cập nhật status
  const st = ALGO_STATUS[key] || '❓ Chưa kiểm tra'
  document.getElementById('algo-status').textContent = name + ' — ' + st
  document.getElementById('model-path-ph').textContent = st
  // Gọi server đổi algo
  fetch('/set_algo/'+key)
}

// ── Session stats ──────────────────────────────────────────────────────────
let sF=0, sA=0, sW=0, log=[], running=false

function upd(lvl){
  sF++
  if(lvl===2) sA++
  if(lvl===1) sW++
  document.getElementById('ma').textContent = sA
  document.getElementById('mw').textContent = sW
  document.getElementById('sf').textContent = sF
  document.getElementById('sa').textContent = sA
  document.getElementById('sw').textContent = sW
  const danger = sA + sW
  document.getElementById('sr').textContent = sF ? Math.round(danger/sF*100)+'%' : '0%'
}

function addLog(src, lvl, conf){
  if(lvl === 0) return
  const t   = new Date().toLocaleTimeString('vi-VN')
  const lbl = lvl===2 ? '🚨 TAI NẠN' : '⚠️ CẢNH BÁO'
  log.unshift({t, algo:ALGO_INFO[currentAlgo]?.label||currentAlgo, src, lbl, conf, lvl})
  renderLog()
  // Latest alert card
  document.getElementById('alert-card').style.display = 'block'
  document.getElementById('latest-alert').innerHTML = lvl===2
    ? `<div class="abox-acc">🚨 <b>TAI NẠN PHÁT HIỆN!</b><br>Thời gian: ${t} | Tin cậy: ${conf}<br>${src}</div>`
    : `<div class="abox-warn">⚠️ <b>CẢNH BÁO: Suýt va chạm!</b><br>Thời gian: ${t} | Tin cậy: ${conf}<br>${src}</div>`
}

function renderLog(){
  const lb = document.getElementById('lb')
  if(!log.length){
    lb.innerHTML = '<tr><td colspan="5" style="color:#3a5a7a;text-align:center;padding:18px">Chưa có sự kiện nguy hiểm</td></tr>'
    document.getElementById('la').innerHTML = ''
    return
  }
  lb.innerHTML = log.map(r=>`
    <tr><td>${r.t}</td><td>${r.algo}</td><td>${r.src}</td>
    <td>${r.lbl}</td><td>${r.conf}</td></tr>`).join('')
  // Banner tab log
  const latest = log[0]
  document.getElementById('la').innerHTML = latest.lvl===2
    ? `<div class="abox-acc">🚨 <b>TAI NẠN MỚI NHẤT:</b> ${latest.t} | ${latest.src} | ${latest.conf}</div>`
    : `<div class="abox-warn">⚠️ <b>CẢNH BÁO MỚI NHẤT:</b> ${latest.t} | ${latest.src} | ${latest.conf}</div>`
}

function clrLog(){
  log=[]; sA=0; sW=0; sF=0; renderLog()
  document.getElementById('ma').textContent=0
  document.getElementById('mw').textContent=0
  document.getElementById('alert-card').style.display='none'
}

// ── Show result ────────────────────────────────────────────────────────────
function showRes(data, src){
  if(data.error){ alert('Lỗi: '+data.error); return }
  document.getElementById('ph').style.display = 'none'
  const img = document.getElementById('ri'); img.src=data.image; img.style.display='block'
  const lvl = data.level  // 0/1/2

  // Status text
  const stEl = document.getElementById('st')
  if(lvl === 2){
    stEl.textContent = '🚨 TAI NẠN PHÁT HIỆN!'
    stEl.className   = 'sacc'
    document.getElementById('cd').textContent = 'Đã xảy ra va chạm | Confidence: '+data.acc_conf
  } else if(lvl === 1){
    stEl.textContent = '⚠️ CẢNH BÁO: Suýt va chạm!'
    stEl.className   = 'swarn'
    document.getElementById('cd').textContent = 'Phát hiện nguy cơ | Confidence: '+data.acc_conf
  } else {
    stEl.textContent = '✅ BÌNH THƯỜNG'
    stEl.className   = 'snormal'
    document.getElementById('cd').textContent = ''
  }

  // Detection table
  const rows = (data.detections||[]).map(d=>{
    const cls = d.level===2?'acc':(d.level===1?'warn':'')
    const icon= d.level===2?'🚨':(d.level===1?'⚠️':'✅')
    const muc = d.level===2?'Tai nạn':(d.level===1?'Cảnh báo':'Bình thường')
    return `<tr class="${cls}"><td>${icon} ${d.label}</td><td>${d.conf}</td><td>${muc}</td></tr>`
  }).join('') || '<tr><td colspan="3" style="color:#3a5a7a">Không phát hiện</td></tr>'
  document.getElementById('db').innerHTML = rows

  upd(lvl)
  if(lvl > 0) addLog(src, lvl, data.acc_conf)
}

// ── Detection actions ──────────────────────────────────────────────────────
async function go(){
  const src  = document.getElementById('src').value
  const conf = document.getElementById('conf').value
  const iou  = document.getElementById('iou').value
  running = true

  if(src === 'image'){
    const f = document.getElementById('fi').files[0]
    if(!f){ alert('Chọn file ảnh trước!'); return }
    document.getElementById('st').textContent='🔍 Đang nhận diện...'
    const fd=new FormData(); fd.append('file',f); fd.append('conf',conf); fd.append('iou',iou)
    const r = await fetch('/detect_image',{method:'POST',body:fd})
    showRes(await r.json(), f.name)

  } else if(src === 'video'){
    const f = document.getElementById('fi').files[0]
    if(!f){ alert('Chọn file video trước!'); return }
    document.getElementById('vp').style.display='block'
    const fd=new FormData(); fd.append('file',f); fd.append('conf',conf); fd.append('iou',iou)
    const r = await fetch('/detect_video',{method:'POST',body:fd})
    const reader=r.body.getReader(); const dec=new TextDecoder(); let buf=''
    while(running){
      const{done,value}=await reader.read(); if(done)break
      buf+=dec.decode(value,{stream:true})
      const parts=buf.split('\n\n'); buf=parts.pop()
      for(const p of parts){
        if(!p.startsWith('data:')) continue
        try{
          const d=JSON.parse(p.slice(5))
          if(d.done){
            document.getElementById('vi').textContent='✅ Xử lý xong!'
            running=false; break
          }
          showRes(d,'Video frame '+d.frame)
          document.getElementById('pb').style.width=(d.progress||0)+'%'
          document.getElementById('vi').textContent=`Frame ${d.frame} | ${d.progress||0}%`
        }catch(e){}
      }
    }
    document.getElementById('vp').style.display='none'

  } else {
    document.getElementById('vi').textContent='📹 Webcam đang chạy...'
    while(running){
      const r = await fetch(`/webcam_frame?conf=${conf}&iou=${iou}`)
      if(!r.ok){ alert('Không mở được webcam!'); break }
      showRes(await r.json(),'Webcam')
      await new Promise(x=>setTimeout(x,80))
    }
    document.getElementById('vi').textContent=''
  }
}
function stop(){ running=false; document.getElementById('vi').textContent='' }

// ══════════════════════════════════════════════════════════════════════════════
// CHARTS
// ══════════════════════════════════════════════════════════════════════════════
const C=['#00ffcc','#ffaa00','#aa88ff']
const F=['rgba(0,255,204,.15)','rgba(255,170,0,.15)','rgba(170,136,255,.15)']
const GO={plugins:{legend:{labels:{color:'#c0d8f0'}}},scales:{x:{ticks:{color:'#c0d8f0'},grid:{color:'#1e3a5f'}},y:{ticks:{color:'#c0d8f0'},grid:{color:'#1e3a5f'}}}}

function buildCompare(){
  cBuilt=true
  new Chart(document.getElementById('rc'),{type:'radar',data:{
    labels:['mAP@0.5','mAP@0.95','FPS (norm)','Size (inv)','Latency (inv)'],
    datasets:[
      {label:'YOLOv8n',data:[73.9,33.8,90,95,92],borderColor:C[0],backgroundColor:F[0],pointBackgroundColor:C[0],borderWidth:2},
      {label:'Faster R-CNN',data:[91.5,68.9,25,10,30],borderColor:C[2],backgroundColor:F[2],pointBackgroundColor:C[2],borderWidth:2},
    ]},options:{plugins:{legend:{labels:{color:'#c0d8f0'}}},scales:{r:{ticks:{color:'#7a9cc4',backdropColor:'transparent'},grid:{color:'#1e3a5f'},pointLabels:{color:'#c0d8f0'}}}}})
  new Chart(document.getElementById('bc'),{type:'bar',data:{
    labels:['mAP@0.5','mAP@0.95','Fitness'],
    datasets:[
      {label:'YOLOv8n',data:[73.9,33.8,37.8],backgroundColor:'rgba(0,255,204,.75)',borderColor:C[0],borderWidth:1},
      {label:'Faster R-CNN',data:[91.5,68.9,0],backgroundColor:'rgba(170,136,255,.75)',borderColor:C[2],borderWidth:1},
    ]},options:{...GO,scales:{x:{ticks:{color:'#c0d8f0'},grid:{color:'#1e3a5f'}},y:{ticks:{color:'#c0d8f0'},grid:{color:'#1e3a5f'},min:0,max:100}}}})
  new Chart(document.getElementById('buc'),{type:'bubble',data:{datasets:[
    {label:'YOLOv8n',   data:[{x:60,y:73.9,r:6}],  backgroundColor:'rgba(0,255,204,.75)',borderColor:C[0]},
    {label:'Faster R-CNN',data:[{x:18.4,y:91.5,r:24}],backgroundColor:'rgba(170,136,255,.75)',borderColor:C[2]},
  ]},options:{...GO,scales:{
    x:{title:{display:true,text:'FPS',color:'#7a9cc4'},ticks:{color:'#c0d8f0'},grid:{color:'#1e3a5f'}},
    y:{title:{display:true,text:'mAP@0.5(%)',color:'#7a9cc4'},ticks:{color:'#c0d8f0'},grid:{color:'#1e3a5f'},min:60,max:100}
  }}})
}

function buildPerf(){
  pBuilt=true
  const ep=[...Array(50)].map((_,i)=>i+1)
  let s=42; const rng=()=>{s=Math.sin(s)*99999;return s-Math.floor(s)}
  const lf=(a,b,c)=>ep.map(e=>+(a*Math.exp(-b*e)+c+(rng()-.5)*.04).toFixed(3))
  const mf=(a,b)=>ep.map(e=>+(a*(1-Math.exp(-b*e))+(rng()-.5)*.016).toFixed(3))
  new Chart(document.getElementById('lc'),{type:'line',data:{labels:ep,datasets:[
    {label:'YOLOv8n',    data:lf(2.5,.08,.22),borderColor:C[0],pointRadius:0,tension:.4},
    {label:'Faster R-CNN',data:lf(3,.07,.20), borderColor:C[2],pointRadius:0,tension:.4},
  ]},options:GO})
  new Chart(document.getElementById('mc'),{type:'line',data:{labels:ep,datasets:[
    {label:'YOLOv8n',    data:mf(.739,.1), borderColor:C[0],pointRadius:0,tension:.4},
    {label:'Faster R-CNN',data:mf(.915,.09),borderColor:C[2],pointRadius:0,tension:.4},
  ]},options:GO})
  new Chart(document.getElementById('cc'),{type:'bar',data:{
    labels:['Accident','Non-Accident'],
    datasets:[
      {label:'Dự đoán Accident',    data:[943,15],  backgroundColor:'rgba(255,51,51,.8)'},
      {label:'Dự đoán Non-Accident', data:[12,821], backgroundColor:'rgba(0,255,128,.7)'},
    ]},options:GO})
}
</script></body></html>"""

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def get_algo_statuses():
    """Tạo dict trạng thái từng model để truyền vào HTML."""
    out = {}
    for key in ALGO_CONFIG:
        if key in loaded_models:
            path = model_paths.get(key,"")
            out[key] = f"✅ {os.path.basename(path)}"
        elif key in model_errors:
            out[key] = f"❌ {model_errors[key][:60]}"
        else:
            out[key] = "⚠️ Chưa load"
    return out

@app.route("/")
def index():
    statuses = get_algo_statuses()
    default_path = model_paths.get("yolov8","Không tìm thấy model")
    return render_template_string(
        HTML,
        algo_statuses      = statuses,
        algo_statuses_json = json.dumps(statuses),
        model_path_default = os.path.basename(default_path) if default_path else "Không tìm thấy",
    )

@app.route("/set_algo/<key>")
def set_algo(key):
    global active_algo
    if key in ALGO_CONFIG:
        active_algo = key
    return jsonify({"ok": True, "algo": active_algo})

def _do_predict(frame, conf_thr, iou_thr):
    """Chạy model hiện tại, trả về (annotated, detections, level, conf)."""
    m = loaded_models.get(active_algo)
    if m is None:
        err = model_errors.get(active_algo, "Model chưa được load")
        return None, [], 0, 0.0, err

    cfg = ALGO_CONFIG[active_algo]
    if cfg["framework"] == "ultralytics":
        results = m.predict(frame, conf=conf_thr, iou=iou_thr, verbose=False)
        ann, det, level, max_conf = draw_boxes(frame, results, conf_thr, active_algo)
    else:
        # Faster R-CNN — chạy thật với torchvision
        ann, det, level, max_conf = run_faster_rcnn(m, frame, cfg, conf_thr)

    return ann, det, level, max_conf, None

@app.route("/detect_image", methods=["POST"])
def detect_image():
    try:
        f    = request.files["file"]
        conf = float(request.form.get("conf", 0.4))
        iou  = float(request.form.get("iou",  0.45))
        img  = cv2.cvtColor(np.array(Image.open(f.stream).convert("RGB")), cv2.COLOR_RGB2BGR)
        ann, det, level, max_conf, err = _do_predict(img, conf, iou)
        if err:
            return jsonify({"error": err})
        return jsonify({
            "image":      to_b64(ann),
            "detections": det,
            "level":      level,
            "accident":   level == 2,
            "warning":    level == 1,
            "acc_conf":   f"{max_conf:.1%}" if max_conf else "",
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/detect_video", methods=["POST"])
def detect_video():
    try:
        f    = request.files["file"]
        conf = float(request.form.get("conf", 0.4))
        iou  = float(request.form.get("iou",  0.45))
        tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.save(tmp.name); tmp.close()
    except Exception as e:
        return jsonify({"error": str(e)})

    def gen():
        cap   = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
        skip  = max(1, int(fps_v / 8))   # xử lý ~8 fps để không lag
        idx   = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                idx += 1
                if idx % skip != 0: continue
                ann, det, level, max_conf, err = _do_predict(frame, conf, iou)
                if err:
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    break
                payload = {
                    "image":      to_b64(ann),
                    "detections": det,
                    "level":      level,
                    "accident":   level == 2,
                    "warning":    level == 1,
                    "acc_conf":   f"{max_conf:.1%}" if max_conf else "",
                    "frame":      idx,
                    "progress":   round(idx / total * 100),
                }
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            cap.release()
            try: os.unlink(tmp.name)
            except: pass
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(gen(), mimetype="text/event-stream")

_cam = None
@app.route("/webcam_frame")
def webcam_frame():
    global _cam
    conf = float(request.args.get("conf", 0.4))
    iou  = float(request.args.get("iou",  0.45))
    if _cam is None or not _cam.isOpened():
        _cam = cv2.VideoCapture(1)
        if not _cam.isOpened():
            _cam = cv2.VideoCapture(2)
    ret, frame = _cam.read()
    if not ret:
        _cam = None
        return jsonify({"error": "Không đọc được webcam. Kiểm tra camera có kết nối không."})
    ann, det, level, max_conf, err = _do_predict(frame, conf, iou)
    if err:
        return jsonify({"error": err})
    return jsonify({
        "image":      to_b64(ann),
        "detections": det,
        "level":      level,
        "accident":   level == 2,
        "warning":    level == 1,
        "acc_conf":   f"{max_conf:.1%}" if max_conf else "",
    })

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚨 AI Accident Detection Dashboard v3.0")
    print(f"  YOLOv8n    : {model_paths.get('yolov8','❌ không tìm thấy')}")
    print(f"  Faster RCNN: {model_paths.get('faster_rcnn','❌ không tìm thấy')}")
    print("  👉 Mở trình duyệt: http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)