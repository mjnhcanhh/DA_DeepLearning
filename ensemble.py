# -*- coding: utf-8 -*-
"""
ensemble.py — Weighted Voting Ensemble cho 3 model accident detection
FIX: normalize weighted_score đúng cách để threshold có ý nghĩa thực
"""

MODEL_WEIGHTS = {
    "ssd":          0.25,
    "yolov12":      0.30,
    "faster_rcnn":  0.45,
}

# Ngưỡng trên NORMALIZED score (0–1, trong đó 1.0 = tất cả model 100% confident)
# Cũ: ENSEMBLE_THRESHOLD=0.55 → không bao giờ đạt vì max score ~0.45
# Fix: dùng 0.30 = chỉ cần Faster R-CNN (0.45) detect với conf≥0.67
ENSEMBLE_THRESHOLD     = 0.30   # weighted vote đủ mạnh
MAJORITY_THRESHOLD     = 2      # hoặc ≥2 model đồng ý (bất kể score)
HIGH_CONF_SINGLE       = 0.75   # hoặc Faster R-CNN rất chắc chắn
WARNING_THRESHOLD      = 0.15   # dưới đây = bình thường


def normalize_conf(conf):
    try:
        return min(1.0, max(0.0, float(conf or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def ensemble_decision(results):
    """
    Tổng hợp kết quả từ nhiều model → 1 quyết định cuối.

    Logic ưu tiên (OR):
      1. votes >= 2: đa số model đồng ý → accident
      2. normalized weighted_score >= ENSEMBLE_THRESHOLD: điểm tổng hợp đủ cao
      3. Faster R-CNN confidence >= HIGH_CONF_SINGLE: model mạnh nhất rất chắc

    normalized weighted_score = sum(conf_i * weight_i) / sum(weight_i của model loaded)
    → nằm trong [0, 1], ngưỡng 0.30 có nghĩa thực sự
    """
    votes = 0
    raw_weighted = 0.0
    total_weight_loaded = 0.0
    model_details = {}

    for key, weight in MODEL_WEIGHTS.items():
        r = results.get(key) or {}
        loaded = bool(r.get("loaded", False)) and not r.get("error")
        accident = bool(r.get("accident", False))
        conf = normalize_conf(r.get("confidence", r.get("max_conf", 0.0)))

        if loaded:
            total_weight_loaded += weight
            if accident:
                votes += 1
                raw_weighted += conf * weight

        model_details[key] = {
            "accident":   accident,
            "confidence": round(conf, 3),
            "weight":     weight,
            "loaded":     loaded,
            "skipped":    bool(r.get("skipped", False)),
            "error":      r.get("error"),
        }

    # Normalize về [0, 1]
    normalized_score = (raw_weighted / total_weight_loaded) if total_weight_loaded > 0 else 0.0

    # Quyết định (OR của 3 điều kiện)
    final_accident = False
    reason = "Khong du bang chung tai nan"
    level = 0

    rcnn_conf = model_details.get("faster_rcnn", {}).get("confidence", 0)

    if votes >= MAJORITY_THRESHOLD:
        final_accident = True
        reason = f"Da so model dong y ({votes}/3 model phat hien tai nan)"
        level = 2
    elif normalized_score >= ENSEMBLE_THRESHOLD:
        final_accident = True
        reason = f"Weighted vote du nguong ({normalized_score:.2f} >= {ENSEMBLE_THRESHOLD})"
        level = 2
    elif rcnn_conf >= HIGH_CONF_SINGLE:
        final_accident = True
        reason = f"Faster R-CNN rat chac chan (conf={rcnn_conf:.2f} >= {HIGH_CONF_SINGLE})"
        level = 2
    elif normalized_score >= WARNING_THRESHOLD:
        reason = f"Canh bao yeu (score={normalized_score:.2f})"
        level = 1

    return {
        "accident":       final_accident,
        "level":          level,
        "ensemble_score": round(normalized_score, 3),
        "votes":          votes,
        "reason":         reason,
        "models":         model_details,
        "weights_used":   {k: v for k, v in MODEL_WEIGHTS.items()},
    }
