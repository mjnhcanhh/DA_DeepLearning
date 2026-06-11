"""
app.py — AI Accident Detection Dashboard v4
Entry point — chứa Flask app + routes
UPDATED: thêm /benchmark_image route để so sánh thật 3 model
"""

import os, json, base64, tempfile, io, time
from collections import deque
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
import cv2, numpy as np
from PIL import Image

from models import loaded_models, model_paths, model_errors, ALGO_CONFIG, active_algo as _init_algo
from detection import draw_boxes, run_faster_rcnn, classify_label, to_b64
from ensemble import ensemble_decision
from camera_map import get_all_cameras, random_source, random_camera, assign_uploaded_to_camera
from emergency import (
    create_incident,
    dispatch_incident,
    get_emergency_units,
    get_feedback_stats,
    get_incidents,
    update_status as set_incident_status,
)

app = Flask(__name__)

# ── active algo state ──────────────────────────────────────────────────────
_current_algo = {"key": _init_algo}
_camera_health = {}

def get_current_algo():
    return _current_algo["key"]

def _format_conf(max_conf):
    return f"{float(max_conf):.2f}" if max_conf else ""

def _format_metric(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{float(value):.1f}{suffix}"

def _verify_accident_with_model(frame, cfg, iou_thr):
    verify_key = cfg.get("verify_with")
    if not verify_key:
        return True
    verifier = loaded_models.get(verify_key)
    verify_cfg = ALGO_CONFIG.get(verify_key, {})
    if verifier is None or verify_cfg.get("framework") != "ultralytics":
        return None

    verify_thr = float(cfg.get("verify_score_thresh", verify_cfg.get("score_thresh", 0.5)) or 0.5)
    allowed_ids = set(cfg.get("verify_allowed_class_ids", verify_cfg.get("allowed_class_ids", [])))
    normal_ids = set(cfg.get("verify_normal_class_ids", []))
    normal_thr = float(cfg.get("verify_normal_score_thresh", 0.7) or 0.7)
    preds = verifier.predict(frame, conf=verify_thr, iou=iou_thr, verbose=False)
    if not preds:
        return False

    boxes = getattr(preds[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False
    saw_normal = False
    for cls_id, score in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
        if allowed_ids and int(cls_id) not in allowed_ids:
            if normal_ids and int(cls_id) in normal_ids and float(score) >= normal_thr:
                saw_normal = True
            continue
        if float(score) >= verify_thr:
            return True
    if saw_normal:
        return "normal"
    return False

def _clear_unverified(frame, cfg, max_conf):
    keep_thr = float(cfg.get("unverified_keep_score", 1.0) or 1.0)
    if max_conf and float(max_conf) >= keep_thr:
        return False
    return True

def _postprocess_cnn_result(frame, cfg, ann, det, level, max_conf, iou_thr):
    if level != 2:
        return ann, det, level, max_conf
    verified = _verify_accident_with_model(frame, cfg, iou_thr)
    if verified == "normal" or _clear_unverified(frame, cfg, max_conf):
        return frame.copy(), [], 0, 0.0
    if verified is True:
        return ann, det, level, max_conf
    return ann, det, level, max_conf

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_algo_statuses():
    out = {}
    for key in ALGO_CONFIG:
        if key in loaded_models:
            path = model_paths.get(key, "")
            out[key] = f"✅ {os.path.basename(path)}" 
        elif key in model_errors:
            out[key] = f"❌ {model_errors[key][:60]}"
        else:
            out[key] = "⚠️ Chưa load"
    return out

def _do_predict(frame, conf_thr, iou_thr):
    algo = get_current_algo()
    m = loaded_models.get(algo)
    if m is None:
        err = model_errors.get(algo, "Model chưa được load")
        return None, [], 0, 0.0, err
    cfg = ALGO_CONFIG[algo]
    if cfg["framework"] == "ultralytics":
        results = m.predict(frame, conf=conf_thr, iou=iou_thr, verbose=False)
        ann, det, level, max_conf = draw_boxes(frame, results, conf_thr, algo, cfg)
    else:
        ann, det, level, max_conf = run_faster_rcnn(m, frame, cfg, conf_thr)
        ann, det, level, max_conf = _postprocess_cnn_result(frame, cfg, ann, det, level, max_conf, iou_thr)
    return ann, det, level, max_conf, None

def _predict_model(frame, key, conf_thr, iou_thr):
    cfg = ALGO_CONFIG[key]
    m = loaded_models.get(key)
    if m is None:
        return {
            "loaded": False,
            "name": cfg["name"],
            "error": model_errors.get(key, "Chua load model"),
            "color": cfg.get("color", "#64748b"),
        }

    t0 = time.perf_counter()
    if cfg["framework"] == "ultralytics":
        preds = m.predict(frame, conf=conf_thr, iou=iou_thr, verbose=False)
        ann, det, level, max_conf = draw_boxes(frame, preds, conf_thr, key, cfg)
    else:
        ann, det, level, max_conf = run_faster_rcnn(m, frame, cfg, conf_thr)
        ann, det, level, max_conf = _postprocess_cnn_result(frame, cfg, ann, det, level, max_conf, iou_thr)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "loaded": True,
        "name": cfg["name"],
        "image": to_b64(ann),
        "detections": det,
        "level": level,
        "accident": level == 2,
        "confidence": round(float(max_conf), 3) if max_conf else 0.0,
        "max_conf": round(float(max_conf), 3) if max_conf else 0.0,
        "acc_conf": _format_conf(max_conf) if max_conf else "",
        "timing_ms": ms,
        "fps": round(1000 / ms, 1) if ms > 0 else 0,
        "num_det": len(det),
        "color": cfg.get("color", "#64748b"),
    }

def _predict_all_models(frame, conf_thr, iou_thr):
    results = {}
    for key in ALGO_CONFIG:
        try:
            results[key] = _predict_model(frame.copy(), key, conf_thr, iou_thr)
        except Exception as e:
            cfg = ALGO_CONFIG[key]
            results[key] = {
                "loaded": key in loaded_models,
                "name": cfg["name"],
                "error": str(e),
                "color": cfg.get("color", "#64748b"),
            }
    return results

def _predict_cascade(frame, conf_thr, iou_thr):
    """
    Fast-first cascade for camera workflows:
    SSD + YOLOv12 run first. Faster R-CNN only verifies suspicious frames.
    """
    t0 = time.perf_counter()
    results = {}
    cascade_steps = []
    fast_keys = [key for key in ("ssd", "yolov12") if key in ALGO_CONFIG]

    for key in fast_keys:
        results[key] = _predict_model(frame.copy(), key, conf_thr, iou_thr)
        if results[key].get("loaded") and not results[key].get("error"):
            cascade_steps.append(key)

    fast_votes = sum(1 for key in fast_keys if results.get(key, {}).get("accident"))
    fast_conf = max((float(results.get(key, {}).get("confidence") or 0.0) for key in fast_keys), default=0.0)
    should_verify = fast_votes > 0 or fast_conf >= max(0.35, float(conf_thr))

    if should_verify and "faster_rcnn" in ALGO_CONFIG:
        results["faster_rcnn"] = _predict_model(frame.copy(), "faster_rcnn", conf_thr, iou_thr)
        if results["faster_rcnn"].get("loaded") and not results["faster_rcnn"].get("error"):
            cascade_steps.append("faster_rcnn")
    elif "faster_rcnn" in ALGO_CONFIG:
        cfg = ALGO_CONFIG["faster_rcnn"]
        results["faster_rcnn"] = {
            "loaded": False,
            "name": cfg["name"],
            "skipped": True,
            "accident": False,
            "confidence": 0.0,
            "max_conf": 0.0,
            "color": cfg.get("color", "#64748b"),
            "reason": "Skipped by cascade: fast models found no accident pattern",
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return results, {
        "mode": "fast_to_verifier",
        "fast_models": fast_keys,
        "verifier": "faster_rcnn",
        "verifier_ran": bool(results.get("faster_rcnn", {}).get("timing_ms")),
        "steps": cascade_steps,
        "fast_votes": fast_votes,
        "fast_conf": round(fast_conf, 3),
        "latency_ms": elapsed_ms,
        "fps": round(1000 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
    }

def _priority_score(ensemble, cascade=None):
    score = float(ensemble.get("ensemble_score") or 0) * 70
    score += min(int(ensemble.get("votes") or 0), 3) / 3 * 20
    if cascade and cascade.get("verifier_ran"):
        score += 10
    return int(max(0, min(100, round(score))))

def _update_camera_health(cam, status, cascade=None, source=None):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    latency = (cascade or {}).get("latency_ms")
    fps = (cascade or {}).get("fps")
    _camera_health[cam["id"]] = {
        "camera_id": cam["id"],
        "camera_name": cam["name"],
        "lat": cam["lat"],
        "lng": cam["lng"],
        "online": status not in {"no_source", "source_error"},
        "status": status,
        "last_scan": now,
        "latency_ms": latency,
        "fps": fps,
        "cascade": cascade or {},
        "source": source,
    }
    return _camera_health[cam["id"]]

def _static_url_for_path(path):
    try:
        p = Path(path).resolve()
        static_root = Path("static").resolve()
        rel = p.relative_to(static_root)
        return "/static/" + rel.as_posix()
    except Exception:
        return None

def _fast_video_model_key():
    for key in ("yolov12", "ssd"):
        if key in loaded_models:
            return key
    return get_current_algo()

def _scan_video_with_fast_model(video_path, conf_thr, iou_thr, max_samples=18):
    key = _fast_video_model_key()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"error": "Khong mo duoc video"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(total / max_samples))
    frame_idx = 0
    samples = 0
    votes = 0
    best = None
    timings = []
    t0_all = time.perf_counter()

    try:
        while frame_idx < total and samples < max_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            samples += 1
            result = _predict_model(frame.copy(), key, conf_thr, iou_thr)
            timings.append(float(result.get("timing_ms") or 0))
            if result.get("accident"):
                votes += 1
            if best is None or float(result.get("confidence") or 0) > float(best.get("confidence") or 0):
                best = result
                best["frame"] = frame_idx
                best["second"] = round(frame_idx / fps_v, 2) if fps_v else 0
            frame_idx += step
    finally:
        cap.release()

    elapsed = round((time.perf_counter() - t0_all) * 1000, 1)
    avg_ms = round(sum(timings) / len(timings), 1) if timings else None
    confirmed = votes >= 2 or (best and best.get("accident") and float(best.get("confidence") or 0) >= 0.65)
    cascade = {
        "mode": "video_fast_yolo",
        "fast_models": [key],
        "verifier": None,
        "verifier_ran": False,
        "steps": [key],
        "samples": samples,
        "votes": votes,
        "best_second": best.get("second") if best else None,
        "latency_ms": elapsed,
        "avg_frame_ms": avg_ms,
        "fps": round(1000 / avg_ms, 1) if avg_ms else 0,
        "video_fps": round(fps_v, 1),
    }

    model_results = {}
    if best:
        model_results[key] = best
    for other in ALGO_CONFIG:
        if other == key:
            continue
        cfg = ALGO_CONFIG[other]
        model_results[other] = {
            "loaded": False,
            "name": cfg["name"],
            "skipped": True,
            "accident": False,
            "confidence": 0.0,
            "max_conf": 0.0,
            "color": cfg.get("color", "#64748b"),
            "reason": "Skipped for uploaded video: fast YOLO/SSD sampling mode",
        }
    if best and confirmed:
        model_results[key]["accident"] = True
        model_results[key]["level"] = 2
    return model_results, cascade

def _read_demo_frame(source):
    """
    Đọc 1 frame từ source (ảnh hoặc video).
    Video: lấy frame NGẪU NHIÊN trong nửa đầu clip (tránh frame đen cuối clip).
    """
    import random as _rng
    path = Path(source)
    if not path.exists():
        return None
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            # Random trong khoảng [10%, 60%] của video để tránh intro/outro tối
            target = _rng.randint(max(0, int(total * 0.10)), max(1, int(total * 0.60)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:  # fallback về frame đầu
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()
    return cv2.imread(str(path))

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    statuses = get_algo_statuses()
    default_path = model_paths.get("ssd", "Không tìm thấy model")
    return render_template(
        "index.html",
        algo_statuses=statuses,
        algo_statuses_json=json.dumps(statuses),
        model_path_default=os.path.basename(default_path) if default_path else "Không tìm thấy",
    )

@app.route("/set_algo/<key>")
def set_algo(key):
    if key in ALGO_CONFIG:
        _current_algo["key"] = key
    return jsonify({"ok": True, "algo": _current_algo["key"]})

@app.route("/detect_image", methods=["POST"])
def detect_image():
    """Nhận 1 file ảnh, trả JSON. Gọi nhiều lần cho multi-upload từ JS."""
    import time as _t
    try:
        f    = request.files["file"]
        conf = float(request.form.get("conf", 0.4))
        iou  = float(request.form.get("iou",  0.45))
        img  = cv2.cvtColor(np.array(Image.open(f.stream).convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        t0   = _t.perf_counter()
        ann, det, level, max_conf, err = _do_predict(img, conf, iou)
        elapsed_ms = round((_t.perf_counter() - t0) * 1000, 1)
        if err:
            return jsonify({"error": err})
        algo = get_current_algo()
        return jsonify({
            "image":      to_b64(ann),
            "detections": det,
            "level":      level,
            "accident":   level == 2,
            "warning":    level == 1,
            "acc_conf":   _format_conf(max_conf),
            "timing_ms":  elapsed_ms,
            "img_size":   f"{w}x{h}",
            "algo":       algo,
            "num_det":    len(det),
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK ROUTE — So sánh thật 3 model trên cùng 1 ảnh
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/benchmark_image", methods=["POST"])
def benchmark_image():
    """
    Nhận 1 ảnh, chạy qua TẤT CẢ các model đã load.
    Trả JSON với kết quả từng model: annotated image, detections, timing, fps, level.
    """
    import time as _t
    try:
        f    = request.files["file"]
        conf = float(request.form.get("conf", 0.4))
        iou  = float(request.form.get("iou",  0.45))
        img_bytes = f.read()
        arr  = np.frombuffer(img_bytes, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Không đọc được ảnh"})
        h, w = img.shape[:2]
    except Exception as e:
        return jsonify({"error": str(e)})

    bench_results = {}
    for key, cfg in ALGO_CONFIG.items():
        m = loaded_models.get(key)
        if m is None:
            bench_results[key] = {
                "error":   model_errors.get(key, "Chưa load model"),
                "loaded":  False,
                "name":    cfg["name"],
                "color":   cfg.get("color", "#64748b"),
            }
            continue
        try:
            frame = img.copy()
            t0    = _t.perf_counter()
            if cfg["framework"] == "ultralytics":
                preds = m.predict(frame, conf=conf, iou=iou, verbose=False)
                ann, det, level, max_conf = draw_boxes(frame, preds, conf, key, cfg)
            else:
                ann, det, level, max_conf = run_faster_rcnn(m, frame, cfg, conf)
                ann, det, level, max_conf = _postprocess_cnn_result(frame, cfg, ann, det, level, max_conf, iou)
            ms = round((_t.perf_counter() - t0) * 1000, 1)

            bench_results[key] = {
                "loaded":      True,
                "name":        cfg["name"],
                "image":       to_b64(ann),
                "detections":  det,
                "level":       level,
                "accident":    level == 2,
                "max_conf":    round(float(max_conf), 3) if max_conf else 0,
                "acc_conf":    _format_conf(max_conf) if max_conf else "—",
                "timing_ms":   ms,
                "fps":         round(1000 / ms, 1) if ms > 0 else 0,
                "num_det":     len(det),
                # Static config info
                "map50":       cfg.get("map50"),
                "map50_str":   _format_metric(cfg.get("map50"), "%"),
                "size":        cfg.get("size", "—"),
                "latency_cfg": cfg.get("latency", "—"),
                "color":       cfg.get("color", "#64748b"),
                "rank":        cfg.get("rank", 99),
            }
        except Exception as e:
            bench_results[key] = {
                "loaded":  True,
                "name":    cfg["name"],
                "error":   str(e),
                "color":   cfg.get("color", "#64748b"),
            }

    return jsonify({
        "results":   bench_results,
        "img_size":  f"{w}x{h}",
        "num_models": sum(1 for r in bench_results.values() if r.get("loaded") and not r.get("error")),
        "timestamp": _t.time(),
    })


@app.route("/benchmark_status")
@app.route("/model_status")
def benchmark_status():
    """Trả về trạng thái load của từng model (để UI biết model nào ready)."""
    out = {}
    for key, cfg in ALGO_CONFIG.items():
        out[key] = {
            "name":    cfg["name"],
            "loaded":  key in loaded_models,
            "error":   model_errors.get(key, None),
            "path":    os.path.basename(model_paths.get(key, "")) if key in model_paths else None,
            "map50":   cfg.get("map50"),
            "fps":     cfg.get("fps"),
            "latency": cfg.get("latency", "—"),
            "size":    cfg.get("size", "—"),
            "color":   cfg.get("color", "#64748b"),
        }
    return jsonify(out)


# Camera map / emergency workflow
@app.route("/cameras")
def cameras():
    return jsonify(get_all_cameras())

@app.route("/camera_health")
def camera_health():
    cameras = get_all_cameras()
    out = []
    for cam in cameras:
        out.append(_camera_health.get(cam["id"], {
            "camera_id": cam["id"],
            "camera_name": cam["name"],
            "lat": cam["lat"],
            "lng": cam["lng"],
            "online": False,
            "status": "not_scanned",
            "last_scan": None,
            "latency_ms": None,
            "fps": None,
            "cascade": {},
        }))
    return jsonify(out)


@app.route("/scan_cameras", methods=["POST"])
def scan_cameras():
    conf = float(request.form.get("conf", request.args.get("conf", 0.4)))
    iou = float(request.form.get("iou", request.args.get("iou", 0.45)))
    scan_results = []

    for cam in get_all_cameras():
        source = random_source(cam["id"])
        if not source:
            health = _update_camera_health(cam, "no_source")
            scan_results.append({
                "camera_id": cam["id"],
                "camera_name": cam["name"],
                "lat": cam["lat"],
                "lng": cam["lng"],
                "accident": False,
                "status": "no_source",
                "message": "Chua co anh/video demo cho camera nay",
                "ensemble_score": 0,
                "votes": 0,
                "incident_id": None,
                "health": health,
            })
            continue

        frame = _read_demo_frame(source)
        if frame is None:
            health = _update_camera_health(cam, "source_error", source=source)
            scan_results.append({
                "camera_id": cam["id"],
                "camera_name": cam["name"],
                "lat": cam["lat"],
                "lng": cam["lng"],
                "accident": False,
                "status": "source_error",
                "message": "Khong doc duoc nguon demo",
                "source": source,
                "ensemble_score": 0,
                "votes": 0,
                "incident_id": None,
                "health": health,
            })
            continue

        model_results, cascade = _predict_cascade(frame, conf, iou)
        ensemble = ensemble_decision(model_results)
        priority = _priority_score(ensemble, cascade)
        best_image = next((r.get("image") for r in model_results.values() if r.get("image")), None)
        ensemble["priority_score"] = priority
        ensemble["cascade"] = cascade
        incident = create_incident(cam, ensemble, best_image, source) if ensemble["accident"] else None
        health = _update_camera_health(cam, "accident" if ensemble["accident"] else "normal", cascade, source)

        scan_results.append({
            "camera_id": cam["id"],
            "camera_name": cam["name"],
            "lat": cam["lat"],
            "lng": cam["lng"],
            "accident": ensemble["accident"],
            "status": "accident" if ensemble["accident"] else "normal",
            "ensemble_score": ensemble["ensemble_score"],
            "votes": ensemble["votes"],
            "priority_score": priority,
            "reason": ensemble["reason"],
            "models": ensemble["models"],
            "cascade": cascade,
            "health": health,
            "source": source,
            "image": best_image,
            "incident_id": incident["id"] if incident else None,
        })

    return jsonify({"results": scan_results, "incidents": get_incidents()})


@app.route("/scan_random_camera_image", methods=["POST"])
def scan_random_camera_image():
    try:
        f = request.files["file"]
        conf = float(request.form.get("conf", request.args.get("conf", 0.4)))
        iou  = float(request.form.get("iou",  request.args.get("iou",  0.45)))
        # Đọc bytes một lần để dùng cả decode lẫn lưu file
        file_bytes = f.read()
        ext = Path(f.filename or "").suffix.lower()
        is_video = ext in {".mp4", ".avi", ".mov", ".mkv"}
        img = None
        if not is_video:
            arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Khong doc duoc anh/video"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    cam = random_camera()
    if not cam:
        return jsonify({"error": "Khong co camera nao de gan anh"}), 400

    # Lưu file vào thư mục demo của camera → lần sau scan_cameras sẽ tìm thấy
    try:
        saved_path = assign_uploaded_to_camera(cam["id"], file_bytes, f.filename or "upload.jpg")
        source = str(saved_path)
    except Exception:
        source = f"uploaded:{f.filename}"

    video_url = _static_url_for_path(source) if is_video else None
    if is_video:
        model_results, cascade = _scan_video_with_fast_model(source, conf, iou)
        if model_results is None:
            return jsonify({"error": cascade.get("error", "Khong xu ly duoc video")}), 400
    else:
        model_results, cascade = _predict_cascade(img, conf, iou)
    ensemble = ensemble_decision(model_results)
    priority = _priority_score(ensemble, cascade)
    best_image = next((r.get("image") for r in model_results.values() if r.get("image")), None)
    ensemble["priority_score"] = priority
    ensemble["cascade"] = cascade
    ensemble["video_url"] = video_url
    incident = create_incident(cam, ensemble, best_image, source) if ensemble["accident"] else None
    health = _update_camera_health(cam, "accident" if ensemble["accident"] else "normal", cascade, source)
    result = {
        "camera_id":      cam["id"],
        "camera_name":    cam["name"],
        "lat":            cam["lat"],
        "lng":            cam["lng"],
        "accident":       ensemble["accident"],
        "status":         "accident" if ensemble["accident"] else "normal",
        "ensemble_score": ensemble["ensemble_score"],
        "votes":          ensemble["votes"],
        "priority_score": priority,
        "reason":         ensemble["reason"],
        "models":         ensemble["models"],
        "cascade":        cascade,
        "health":         health,
        "video_url":      video_url,
        "source":         source,
        "image":          best_image,
        "incident_id":    incident["id"] if incident else None,
        "message":        f"{'Video' if is_video else 'Anh'} da duoc luu vao {cam['id']} va chay AI",
    }
    return jsonify({"result": result, "incidents": get_incidents()})


@app.route("/map_view")
def map_view():
    """Trang bản đồ riêng (có thể nhúng iframe hoặc mở tab mới)."""
    statuses = get_algo_statuses()
    return render_template(
        "map_view.html",
        cameras=get_all_cameras(),
        algo_statuses_json=json.dumps(statuses),
    )


@app.route("/incidents")
def incidents():
    return jsonify(get_incidents())

@app.route("/feedback_stats")
def feedback_stats():
    return jsonify(get_feedback_stats())


@app.route("/incident/<incident_id>/status", methods=["POST"])
def update_incident(incident_id):
    payload = request.get_json(silent=True) or request.form
    incident = set_incident_status(incident_id, payload.get("status"))
    if not incident:
        return jsonify({"error": "Khong tim thay incident hoac status khong hop le"}), 404
    return jsonify({"ok": True, "incident": incident})


@app.route("/incident/<incident_id>/dispatch", methods=["POST"])
def dispatch_incident_route(incident_id):
    incident = dispatch_incident(incident_id)
    if not incident:
        return jsonify({"error": "Khong tim thay incident"}), 404
    return jsonify({"ok": True, "incident": incident})


@app.route("/emergency_units")
def emergency_units():
    return jsonify(get_emergency_units())


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO / WEBCAM ROUTES (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════════════

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
        skip  = max(1, int(fps_v / 8))
        idx   = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                idx += 1
                if idx % skip != 0: continue
                ann, det, level, max_conf, err = _do_predict(frame, conf, iou)
                if err:
                    yield f"data: {json.dumps({'error': err})}\n\n"; break
                payload = {
                    "image":      to_b64(ann),
                    "detections": det,
                    "level":      level,
                    "accident":   level == 2,
                    "warning":    level == 1,
                    "acc_conf":   _format_conf(max_conf),
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
_webcam_ai = {
    "history": deque(maxlen=10),
    "frame_no": 0,
    "last_incident_ts": 0.0,
}

def _reset_webcam_ai():
    _webcam_ai["history"].clear()
    _webcam_ai["frame_no"] = 0
    _webcam_ai["last_incident_ts"] = 0.0

def _update_webcam_ai(level, max_conf, det_count):
    _webcam_ai["frame_no"] += 1
    is_accident = level == 2
    is_warning = level == 1
    conf = float(max_conf or 0.0)
    _webcam_ai["history"].append({
        "accident": is_accident,
        "warning": is_warning,
        "conf": conf,
        "det_count": int(det_count or 0),
    })

    hist = list(_webcam_ai["history"])
    total = len(hist)
    votes = sum(1 for item in hist if item["accident"])
    warn_votes = sum(1 for item in hist if item["warning"])
    avg_conf = sum(item["conf"] for item in hist) / total if total else 0.0
    vote_ratio = votes / total if total else 0.0
    risk_score = min(100, round((vote_ratio * 70) + (avg_conf * 30)))
    confirmed = total >= 6 and (votes >= 4 or (votes >= 3 and avg_conf >= 0.75))
    suspect = not confirmed and total >= 4 and (votes >= 2 or warn_votes >= 3)

    if confirmed:
        status = "CONFIRMED_ACCIDENT"
        severity = "CRITICAL" if risk_score >= 75 else "ACCIDENT"
        message = "Confirmed by temporal voting"
    elif suspect:
        status = "SUSPECT"
        severity = "WARNING"
        message = "Suspect pattern, waiting for more frames"
    else:
        status = "NORMAL"
        severity = "NORMAL"
        message = "No stable accident pattern"

    return {
        "enabled": True,
        "frame_no": _webcam_ai["frame_no"],
        "window": total,
        "votes": votes,
        "warn_votes": warn_votes,
        "required_votes": 4,
        "avg_conf": round(avg_conf, 3),
        "risk_score": risk_score,
        "confirmed": confirmed,
        "suspect": suspect,
        "status": status,
        "severity": severity,
        "message": message,
    }

def _maybe_create_webcam_incident(temporal, image_b64):
    if not temporal.get("confirmed"):
        return None
    now = time.time()
    if now - _webcam_ai["last_incident_ts"] < 20:
        return None
    cam = random_camera() or {
        "id": "LIVE-WEBCAM",
        "name": "Live Webcam",
        "lat": 10.7769,
        "lng": 106.7009,
    }
    ensemble_like = {
        "accident": True,
        "ensemble_score": temporal["risk_score"],
        "votes": temporal["votes"],
        "reason": f"Live temporal voting: {temporal['votes']}/{temporal['window']} frames",
        "models": [{
            "name": ALGO_CONFIG.get(get_current_algo(), {}).get("name", get_current_algo()),
            "accident": True,
            "confidence": temporal["avg_conf"],
        }],
    }
    incident = create_incident(cam, ensemble_like, image_b64, "live-webcam")
    _webcam_ai["last_incident_ts"] = now
    return incident

@app.route("/webcam_temporal_reset", methods=["POST"])
def webcam_temporal_reset():
    _reset_webcam_ai()
    return jsonify({"ok": True})

@app.route("/webcam_frame")
def webcam_frame():
    global _cam
    conf = float(request.args.get("conf", 0.4))
    iou  = float(request.args.get("iou",  0.45))
    if _cam is None or not _cam.isOpened():
        _cam = cv2.VideoCapture(0)
        if not _cam.isOpened():
            _cam = cv2.VideoCapture(1)
    ret, frame = _cam.read()
    if not ret:
        _cam = None
        return jsonify({"error": "Không đọc được webcam. Kiểm tra camera có kết nối không."})
    ann, det, level, max_conf, err = _do_predict(frame, conf, iou)
    if err:
        return jsonify({"error": err})
    image_b64 = to_b64(ann)
    temporal = _update_webcam_ai(level, max_conf, len(det))
    incident = _maybe_create_webcam_incident(temporal, image_b64)
    return jsonify({
        "image":      image_b64,
        "detections": det,
        "level":      level,
        "accident":   level == 2,
        "warning":    level == 1,
        "acc_conf":   _format_conf(max_conf),
        "temporal":   temporal,
        "incident_id": incident["id"] if incident else None,
    })

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    statuses = get_algo_statuses()
    print("\n" + "="*55)
    print("  🚨 AI Accident Detection Dashboard v4.0")
    print(f"  SSD        : {model_paths.get('ssd','❌ không tìm thấy')}")
    print(f"  Faster RCNN: {model_paths.get('faster_rcnn','❌ không tìm thấy')}")
    print(f"  YOLOv12    : {model_paths.get('yolov12','❌ không tìm thấy')}")
    print("  👉 Mở trình duyệt: http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
