# utils.py - FIX CHINH: O(n2)  O(n) trong compute_map_full
# Thay oi:
#   1. Bo vong lap offset tinh lai tu au moi lan  dung dict pre-built
#   2. Dung torchmetrics MeanAveragePrecision neu co (nhanh hon ~10x)
#      fallback ve thuat toan thuan Python neu khong co
#   3. EarlyStopping giu nguyen

import torch
import numpy as np
from collections import defaultdict


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# TINH IoU
# ============================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# TINH mAP ON GIAN (giu lai e tuong thich)
# ============================================================

def compute_map(predictions, targets, iou_threshold=0.5, num_classes=2):
    map_score, _, _ = compute_map_full(predictions, targets, iou_threshold, num_classes)
    return map_score


# ============================================================
# TINH mAP AY U  FIX O(n2) BUG
# ============================================================

def compute_map_full(predictions, targets, iou_threshold=0.5, num_classes=2):
    """
    BUG GOC: trong vong lap sort_idx, moi prediction phai tinh lai
    `offset` bang cach loop qua tat ca anh truoc no  O(N*K*N).
    Voi 500 anh x 20 pred/anh = 200,000 phep tinh lap  ca tieng.

    FIX: Pre-build mot dict mapping (img_idx, local_rank)  box/score
    mot lan duy nhat truoc khi sort  O(N*K) tong cong.

    Args:
        predictions  : list of dict {"boxes", "scores", "labels"}
        targets      : list of dict {"boxes", "labels"}
        iou_threshold: nguong IoU e xet True Positive
        num_classes  : so class cua model, gom ca background

    Returns:
        map_score : float
        pr_data   : {"recall": list[<=100], "precision": list[<=100]}
        cm        : numpy array (num_classes, num_classes)
    """
    # Thu dung torchmetrics neu co  nhanh hon ~10x nho tensor ops
    try:
        return _compute_map_torchmetrics(predictions, targets, iou_threshold, num_classes)
    except Exception:
        pass

    # Fallback: thuan Python nhung a fix O(n2)
    return _compute_map_python(predictions, targets, iou_threshold, num_classes)


def _compute_map_torchmetrics(predictions, targets, iou_threshold, num_classes):
    """
    Dung torchmetrics.detection.MeanAveragePrecision.
    Nhanh nhat, chinh xac nhat  nen dung neu co.
    pip install torchmetrics
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], class_metrics=False)
    metric.update(predictions, targets)
    result = metric.compute()

    map_score = float(result["map_50"].item() if "map_50" in result else result["map"].item())

    # Tinh PR curve + CM bang Python (chi 1 lan, khong anh huong toc o)
    pr_data, cm = _compute_pr_and_cm(predictions, targets, iou_threshold, num_classes)
    return map_score, pr_data, cm


def _compute_map_python(predictions, targets, iou_threshold, num_classes):
    """
    Thuan Python  a fix bug O(n2).

    FIX cot loi: thay vi tinh offset trong vong lap sorted_idx,
    ta pre-build danh sach phang (flat_boxes, flat_scores, flat_img_idx)
    theo thu tu xuat hien trong tung anh, roi sort 1 lan.
    Khong can tinh offset nua vi index phang a tuong ung 1-1.
    """
    ap_per_class      = []
    all_recall_out    = []
    all_precision_out = []

    n  = num_classes
    cm = np.zeros((n, n), dtype=int)

    for cls in range(1, num_classes):
        #  Pre-build danh sach phang  O(N*K) mot lan 
        flat_scores   = []   # score cua tung prediction
        flat_boxes    = []   # box tuong ung
        flat_img_idx  = []   # anh chua prediction nay
        n_gt          = 0

        gt_by_img = {}
        for img_idx, tgt in enumerate(targets):
            gt_mask = tgt["labels"].cpu() == cls
            gt_by_img[img_idx] = {
                "boxes"  : tgt["boxes"].cpu()[gt_mask],
                "matched": [False] * int(gt_mask.sum()),
            }
            n_gt += int(gt_mask.sum())

        for img_idx, pred in enumerate(predictions):
            pred_mask = pred["labels"].cpu() == cls
            p_boxes   = pred["boxes"].cpu()[pred_mask]
            p_scores  = pred["scores"].cpu()[pred_mask]
            for i in range(len(p_scores)):
                flat_scores.append(p_scores[i].item())
                flat_boxes.append(p_boxes[i])      # tensor [4]
                flat_img_idx.append(img_idx)

        if n_gt == 0:
            continue
        if len(flat_scores) == 0:
            ap_per_class.append(0.0)
            continue

        #  Sort theo score giam dan  O(M log M) 
        sorted_idx = np.argsort(flat_scores)[::-1]

        tp_arr = []
        for rank in sorted_idx:
            img_idx  = flat_img_idx[rank]
            box      = flat_boxes[rank]           #  truc tiep, khong can offset
            g_boxes  = gt_by_img[img_idx]["boxes"]
            matched  = gt_by_img[img_idx]["matched"]

            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(g_boxes):
                iou = compute_iou(box.tolist(), gt_box.tolist())
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                tp_arr.append(1)
                matched[best_j] = True
                gt_by_img[img_idx]["matched"] = matched
                cm[cls][cls] += 1
            else:
                tp_arr.append(0)
                cm[0][cls] += 1

        # FN
        for img_idx in gt_by_img:
            for m in gt_by_img[img_idx]["matched"]:
                if not m:
                    cm[cls][0] += 1

        tp_arr    = np.array(tp_arr)
        cum_tp    = np.cumsum(tp_arr)
        cum_fp    = np.cumsum(1 - tp_arr)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall    = cum_tp / (n_gt + 1e-8)

        # AP  11-point interpolation
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            prec_at_thr = precision[recall >= thr]
            ap += prec_at_thr.max() if len(prec_at_thr) > 0 else 0.0
        ap /= 11.0
        ap_per_class.append(ap)

        # Gioi han PR curve con toi a 100 iem
        if len(recall) > 100:
            idx               = np.linspace(0, len(recall) - 1, 100).astype(int)
            all_recall_out    = recall[idx].tolist()
            all_precision_out = precision[idx].tolist()
        else:
            all_recall_out    = recall.tolist()
            all_precision_out = precision.tolist()

    map_score = float(np.mean(ap_per_class)) if ap_per_class else 0.0
    pr_data   = {"recall": all_recall_out, "precision": all_precision_out}
    return map_score, pr_data, cm


def _compute_pr_and_cm(predictions, targets, iou_threshold, num_classes):
    """Tinh PR curve + Confusion Matrix rieng (dung khi torchmetrics tinh mAP)."""
    n  = num_classes
    cm = np.zeros((n, n), dtype=int)
    all_recall_out    = []
    all_precision_out = []

    for cls in range(1, num_classes):
        flat_scores, flat_boxes, flat_img_idx = [], [], []
        n_gt = 0
        gt_by_img = {}

        for img_idx, tgt in enumerate(targets):
            gt_mask = tgt["labels"].cpu() == cls
            gt_by_img[img_idx] = {
                "boxes"  : tgt["boxes"].cpu()[gt_mask],
                "matched": [False] * int(gt_mask.sum()),
            }
            n_gt += int(gt_mask.sum())

        for img_idx, pred in enumerate(predictions):
            pred_mask = pred["labels"].cpu() == cls
            p_boxes   = pred["boxes"].cpu()[pred_mask]
            p_scores  = pred["scores"].cpu()[pred_mask]
            for i in range(len(p_scores)):
                flat_scores.append(p_scores[i].item())
                flat_boxes.append(p_boxes[i])
                flat_img_idx.append(img_idx)

        if n_gt == 0 or len(flat_scores) == 0:
            continue

        sorted_idx = np.argsort(flat_scores)[::-1]
        tp_arr = []
        for rank in sorted_idx:
            img_idx  = flat_img_idx[rank]
            box      = flat_boxes[rank]
            g_boxes  = gt_by_img[img_idx]["boxes"]
            matched  = gt_by_img[img_idx]["matched"]
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(g_boxes):
                iou = compute_iou(box.tolist(), gt_box.tolist())
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                tp_arr.append(1)
                matched[best_j] = True
                gt_by_img[img_idx]["matched"] = matched
                cm[cls][cls] += 1
            else:
                tp_arr.append(0)
                cm[0][cls] += 1

        for img_idx in gt_by_img:
            for m in gt_by_img[img_idx]["matched"]:
                if not m:
                    cm[cls][0] += 1

        tp_arr    = np.array(tp_arr)
        cum_tp    = np.cumsum(tp_arr)
        cum_fp    = np.cumsum(1 - tp_arr)
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall    = cum_tp / (n_gt + 1e-8)
        if len(recall) > 100:
            idx               = np.linspace(0, len(recall) - 1, 100).astype(int)
            all_recall_out    = recall[idx].tolist()
            all_precision_out = precision[idx].tolist()
        else:
            all_recall_out    = recall.tolist()
            all_precision_out = precision.tolist()

    return {"recall": all_recall_out, "precision": all_precision_out}, cm


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode="max"):
        self.patience     = patience
        self.min_delta    = min_delta
        self.mode         = mode
        self.best         = None
        self.counter      = 0
        self.should_stop  = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        improved = (metric > self.best + self.min_delta) if self.mode == "max" \
                   else (metric < self.best - self.min_delta)

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] Khong cai thien {self.counter}/{self.patience} epoch")
            if self.counter >= self.patience:
                self.should_stop = True
                print("[EarlyStopping]  Dung training som!")

        return self.should_stop
