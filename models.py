"""
models.py — Cấu hình & load tất cả models AI
Tách riêng để app.py gọn hơn
FIXED:
  - SSD ưu tiên checkpoint accident, không load nhầm yolov12.pt
  - Faster R-CNN tự detect kiến trúc (Standard vs BatchNorm) + num_classes từ checkpoint
  - strict=False + fallback để load được mọi biến thể .pth
"""

import os
from pathlib import Path

# Thư mục gốc của project (nơi chứa app.py / models.py)
BASE_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH CÁC THUẬT TOÁN
# ══════════════════════════════════════════════════════════════════════════════
ALGO_CONFIG = {
    "ssd": {
        "name":        "SSD",
        "label":       "SSD (Accident detector)",
        "map50":       None,
        "fps":         None,
        "latency":     "N/A",
        "size":        "22 MB",
        "rank":        3,
        # ← ưu tiên file đúng tên trước, tránh load nhầm yolov12.pt
        "candidates":  [
            "runs/detect/train8/weights/last.pt",
            "runs/detect/train8/weights/best.pt",
            "Models/ssd_accident.pt",
            "ssd_accident.pt",
            "accident_best.pt",
            "Models/accident_best.pt",
            "best.pt",
            "Models/best.pt",
        ],
        "framework":   "ultralytics",
        "color":       "#2563eb",
        "score_thresh": 0.15,
        "max_det":      1,
        "num_classes": 1,
        "class_names": {0: "accident"},
        "allowed_class_ids": [0],
        "allowed_labels": ["accident"],
        # Tag để _resolve_candidates KHÔNG auto-scan vào file yolov12
        "_exclude_patterns": ["yolov12"],
    },
    "faster_rcnn": {
        "name":        "Faster R-CNN",
        "label":       "Faster R-CNN ResNet50+FPN",
        "map50":       63.8,
        "fps":         7.4,
        "latency":     "135ms",
        "size":        "167 MB",
        "rank":        2,
        "candidates":  [
            "Models/faster_rcnn_accident.pth",
            "faster_rcnn_accident.pth",
            "Models/faster_rcnn_accident.pt",
            "faster_rcnn_accident.pt",
            "Models/fast_cnn_model.pth",
            "fast_cnn_model.pth",
        ],
        "framework":   "torchvision",
        "color":       "#7c3aed",
        "input_size":   640,
        "score_thresh": 0.60,
        "max_det":      1,
        "nms_thresh":   0.20,
        "verify_with":  "yolov12",
        "verify_score_thresh": 0.50,
        "verify_allowed_class_ids": [0],
        "verify_normal_class_ids": [1],
        "verify_normal_score_thresh": 0.70,
        "unverified_keep_score": 0.70,
        "confidence_format": "decimal",
        # num_classes sẽ bị ghi đè tự động khi load từ checkpoint
        "num_classes": 2,
        "class_names": {0: "background", 1: "accident"},
        "_include_patterns": ["faster", "fast_cnn", "rcnn"],
        "_exclude_patterns": ["yolo", "ssd"],
    },
    "yolov12": {
        "name":        "YOLOv12",
        "label":       "YOLOv12 (Mới nhất — Attention)",
        "map50":       None,
        "fps":         None,
        "latency":     "N/A",
        "size":        "6 MB",
        "rank":        1,
        "candidates":  [
            "Models/yolov12.pt",
            "yolov12.pt",
            "Models/yolov12n.pt",
            "yolov12n.pt",
            "Models/yolov12s.pt",
            "yolov12s.pt",
        ],
        "framework":   "ultralytics",
        "color":       "#059669",
        "score_thresh": 0.50,
        "max_det":      1,
        "num_classes": 1,
        "class_names": {0: "accident"},
        "allowed_class_ids": [0],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ULTRALYTICS (SSD / YOLOv12)
# ══════════════════════════════════════════════════════════════════════════════

def try_load_ultralytics(path: str):
    import torch

    # PyTorch 2.6+: allowlist ultralytics classes để dùng weights_only=True
    try:
        import ultralytics.nn.modules
        import ultralytics.nn.tasks
        safe_classes = []
        for mod in [ultralytics.nn.modules, ultralytics.nn.tasks]:
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type):
                    safe_classes.append(obj)
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(safe_classes)
    except Exception:
        pass

    # Patch torch.load → weights_only=False (fallback an toàn cho file nội bộ)
    _orig_load = torch.load
    def _patched_load(f, *args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(f, *args, **kwargs)

    torch.load = _patched_load
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    finally:
        torch.load = _orig_load   # luôn restore dù có lỗi


# ══════════════════════════════════════════════════════════════════════════════
# LOAD FASTER R-CNN — tự detect kiến trúc + num_classes từ checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def _detect_rcnn_config(state: dict) -> tuple[int, bool, bool]:
    """
    Phân tích state_dict để xác định:
      - num_classes  : đọc từ shape của cls_score
      - use_bn_fpn   : FPN có BatchNorm không (inner_blocks.X.1.weight tồn tại)
      - use_bn_head  : box_head dùng Conv+BN thay vì TwoMLPHead
    """
    # --- num_classes ---
    cls_key = "roi_heads.box_predictor.cls_score.weight"
    num_classes = 2  # default
    if cls_key in state:
        num_classes = state[cls_key].shape[0]

    # --- FPN BatchNorm ---
    use_bn_fpn = any(
        "backbone.fpn.inner_blocks.0.1.weight" == k for k in state
    )

    # --- box_head kiểu Conv+BN (Sequential) hay TwoMLPHead (fc6/fc7) ---
    use_bn_head = any(
        k.startswith("roi_heads.box_head.0.") for k in state
    )

    return num_classes, use_bn_fpn, use_bn_head


def _build_rcnn_standard(num_classes: int):
    """Kiến trúc chuẩn torchvision: TwoMLPHead, FPN không BN."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def _build_rcnn_resnet50_v2(num_classes: int):
    """Exact Fast CNN/Faster R-CNN v2 architecture used by the training script."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    model.roi_heads.fg_iou_thresh = 0.4
    model.roi_heads.bg_iou_thresh = 0.3
    model.roi_heads.positive_fraction = 0.5
    model.roi_heads.batch_size_per_image = 256
    model.roi_heads.nms_thresh = 0.3
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.detections_per_img = 20
    return model


def _build_rcnn_bn(num_classes: int):
    """
    Kiến trúc có BatchNorm trong FPN + box_head dạng Sequential Conv+BN.
    Khớp với các checkpoint train bằng mmdetection / custom torchvision có BN.
    """
    import torch
    import torch.nn as nn
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    # Build backbone với BN-enabled FPN
    # torchvision >= 0.13 hỗ trợ norm_layer param trong resnet_fpn_backbone
    try:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            norm_layer=nn.BatchNorm2d,
        )
    except TypeError:
        # torchvision cũ không có norm_layer → fallback chuẩn
        backbone = resnet_fpn_backbone("resnet50", weights=None)

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        backbone=backbone,
    ) if False else _build_rcnn_standard(num_classes)  # placeholder

    # Thực ra build thẳng từ backbone
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_gen = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
    )
    return model


def try_load_torchvision(path: str, cfg_num_classes: int):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    # Một số checkpoint wrap trong dict
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict", "model_state"):
            if key in state:
                state = state[key]
                break

    # Tự phát hiện cấu hình từ checkpoint
    num_classes, use_bn_fpn, use_bn_head = _detect_rcnn_config(state)
    print(f"   [Faster R-CNN] detect → num_classes={num_classes}, bn_fpn={use_bn_fpn}, bn_head={use_bn_head}")

    # Cập nhật class_names trong ALGO_CONFIG theo num_classes thật
    _update_rcnn_class_names(num_classes)

    # Thử build đúng kiến trúc trước
    errors = []
    builders = []

    builders = [
        ("ResNet50 FPN v2 architecture", lambda: _build_rcnn_resnet50_v2(num_classes)),
    ]
    if use_bn_fpn or use_bn_head:
        builders.extend([
            ("BN architecture", lambda: _build_rcnn_bn(num_classes)),
            ("Standard architecture", lambda: _build_rcnn_standard(num_classes)),
        ])
    else:
        builders.extend([
            ("Standard architecture", lambda: _build_rcnn_standard(num_classes)),
            ("BN architecture", lambda: _build_rcnn_bn(num_classes)),
        ])

    for desc, build_fn in builders:
        try:
            model = build_fn()
            missing, unexpected = model.load_state_dict(state, strict=False)
            # Chỉ báo lỗi nếu thiếu key quan trọng (không phải BN auxiliary)
            critical_missing = [k for k in missing if "num_batches_tracked" not in k]
            if critical_missing:
                print(f"   [Faster R-CNN] {desc}: {len(critical_missing)} critical missing keys")
                # Vẫn dùng nếu không có lựa chọn nào tốt hơn
            else:
                print(f"   [Faster R-CNN] {desc}: OK (missing={len(missing)}, unexpected={len(unexpected)})")
            model.eval()
            return model
        except Exception as e:
            errors.append(f"{desc}: {e}")
            continue

    raise RuntimeError("Không load được Faster R-CNN. Errors:\n" + "\n".join(errors))


def try_load_ssd(path: str, num_classes: int):
    import torch
    from torchvision.models.detection import ssd300_vgg16

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict", "model_state"):
            if key in state:
                state = state[key]
                break

    model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"   [SSD] loaded state (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model


def _update_rcnn_class_names(num_classes: int):
    """Cập nhật class_names trong ALGO_CONFIG khớp với num_classes thật."""
    defaults = {
        2: {0: "background", 1: "accident"},
        3: {0: "background", 1: "accident", 2: "near_miss"},
        4: {0: "background", 1: "accident", 2: "near_miss", 3: "vehicle"},
    }
    cfg = ALGO_CONFIG["faster_rcnn"]
    cfg["num_classes"] = num_classes
    if num_classes in defaults:
        cfg["class_names"] = defaults[num_classes]
    else:
        cfg["class_names"] = {i: f"class_{i}" for i in range(num_classes)}
        cfg["class_names"][0] = "background"


# ══════════════════════════════════════════════════════════════════════════════
# RESOLVE FILE CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_candidates(candidates: list, exclude_patterns: list = None, include_patterns: list = None) -> list:
    """
    Trả về danh sách path tuyệt đối từ candidates.
    Thử cả path tương đối (từ CWD) lẫn tương đối từ BASE_DIR.
    Auto-scan thư mục Models/ nhưng lọc bỏ file theo exclude_patterns.
    """
    exclude_patterns = exclude_patterns or []
    include_patterns = include_patterns or []
    result = []

    for c in candidates:
        result.append(Path(c))           # relative to CWD
        result.append(BASE_DIR / c)      # relative to project root

    # Auto-scan Models/
    models_dir = BASE_DIR / "Models"
    if models_dir.exists():
        for ext in ("*.pt", "*.pth"):
            for p in models_dir.glob(ext):
                # Bỏ qua nếu tên file chứa pattern bị loại trừ
                if any(pat.lower() in p.name.lower() for pat in exclude_patterns):
                    continue
                if include_patterns and not any(pat.lower() in p.name.lower() for pat in include_patterns):
                    continue
                result.append(p)

    # Loại trùng, giữ thứ tự
    seen, out = set(), []
    for p in result:
        try:
            key_str = str(p.resolve())
        except Exception:
            key_str = str(p)
        if key_str not in seen:
            seen.add(key_str)
            out.append(p)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: LOAD TẤT CẢ MODELS
# ══════════════════════════════════════════════════════════════════════════════
loaded_models  = {}
model_paths    = {}
model_errors   = {}

for key, cfg in ALGO_CONFIG.items():
    exclude = cfg.get("_exclude_patterns", [])
    include = cfg.get("_include_patterns", [])
    candidates = _resolve_candidates(cfg["candidates"], exclude_patterns=exclude, include_patterns=include)
    found = next((c for c in candidates if c.exists()), None)

    if found:
        found_str = str(found.resolve())
        model_paths[key] = found_str
        try:
            if cfg["framework"] == "ultralytics":
                loaded_models[key] = try_load_ultralytics(found_str)
            else:
                loaded_models[key] = try_load_torchvision(found_str, cfg["num_classes"])
            print(f"✅ [{cfg['name']}] loaded: {found_str}")
        except Exception as e:
            model_errors[key] = str(e)
            print(f"❌ [{cfg['name']}] load error: {e}")
    else:
        scanned = BASE_DIR / "Models"
        model_errors[key] = (
            f"Không tìm thấy file model cho {cfg['name']} "
            f"(đã scan: {scanned})"
        )
        print(f"⚠️  [{cfg['name']}] không có file trong {scanned}")

# Default active algorithm
active_algo = "ssd"


def get_model_info(key: str) -> dict:
    """Trả về thông tin model hiện tại để hiển thị trên UI"""
    cfg = ALGO_CONFIG.get(key, {})
    return {
        "name":    cfg.get("name", key),
        "map50":   cfg.get("map50", 0),
        "fps":     cfg.get("fps", 0),
        "latency": cfg.get("latency", "—"),
        "size":    cfg.get("size", "—"),
        "loaded":  key in loaded_models,
    }
