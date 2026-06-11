# model.py - FIX CLASS IMBALANCE + TUNING ROI HEAD
# Fix 1: fg/bg iou threshold  de match positive sample hon
# Fix 2: positive_fraction 0.250.5  tang ti le positive trong moi batch ROI
# Fix 3: batch_size_per_image 512256  giam so negative sample tuyet oi
# Fix 4: bo custom anchor generator (gay loi IndexError voi FPN v2)

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torch


# ============================================================
# OPTIMIZER  LR tach rieng backbone / head
# ============================================================
def get_optimizer(model, lr_backbone=1e-4, lr_head=1e-3,
                  momentum=0.9, weight_decay=5e-4):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.SGD(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ],
        momentum=momentum,
        weight_decay=weight_decay,
    )


# ============================================================
# MODEL
# ============================================================
def get_model(num_classes, backbone="resnet50_v2", trainable_layers=3):
    """
    Args:
        num_classes      : so class ke ca background (= 0)
        backbone         : "resnet50_v2" (khuyen nghi) | "resnet50" | "mobilenet"
        trainable_layers : 0=freeze all | 3=unfreeze 3 layer cuoi | 5=train all
    """

    # Dung anchor mac inh cua model  custom anchor gay IndexError voi FPN v2
    if backbone == "resnet50_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=trainable_layers,
        )

    elif backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=trainable_layers,
        )

    elif backbone == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT",
            trainable_backbone_layers=trainable_layers,
        )

    else:
        raise ValueError(f"Backbone '{backbone}' khong uoc ho tro.")

    # --- Thay head phan loai cho ung so class ---
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #  FIX CLASS IMBALANCE 
    # Van e: loss_cls = 0.532 sau 31 epoch khong giam
    # Nguyen nhan: ti le negative >> positive  model thien ve predict background

    # FIX 1: bg_iou_thresh mac inh = 0.5 la bug lon nhat
    # Box co IoU 0.3-0.5 bi bo qua hoan toan  mat nhieu positive sample
    model.roi_heads.fg_iou_thresh = 0.4   # cu: 0.5
    model.roi_heads.bg_iou_thresh = 0.3   # cu: 0.5

    # FIX 2: Tang ti le positive trong mini-batch ROI
    # 0.25  128 pos + 384 neg (qua it positive)
    # 0.5   128 pos + 128 neg (can bang hon)
    model.roi_heads.positive_fraction = 0.5   # cu: 0.25

    # FIX 3: Giam batch size e giam so negative tuyet oi
    # 512  256: so positive giu nguyen, negative giam 3x
    model.roi_heads.batch_size_per_image = 256   # cu: 512

    #  TUNING THRESHOLDS 
    model.roi_heads.nms_thresh          = 0.3
    model.roi_heads.score_thresh        = 0.05
    model.roi_heads.detections_per_img  = 20

    return model


# ============================================================
# THONG TIN MODEL
# ============================================================
def print_model_info(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Tong tham so   : {total:,}")
    print(f"[Model] Tham so train  : {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"[Model] Tham so frozen : {total - trainable:,}")
    print(f"[Model] ROI fg_iou     : {model.roi_heads.fg_iou_thresh}")
    print(f"[Model] ROI bg_iou     : {model.roi_heads.bg_iou_thresh}")
    print(f"[Model] ROI pos_frac   : {model.roi_heads.positive_fraction}")
    print(f"[Model] ROI batch/img  : {model.roi_heads.batch_size_per_image}")