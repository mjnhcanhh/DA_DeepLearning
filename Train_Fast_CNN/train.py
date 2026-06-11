# train.py - fixed training pipeline v3
# Fix 1: torch.amp thay torch.cuda.amp (deprecated)
# Fix 2: deduplicate dataset pairs
# Fix 3: batch_size 8, num_workers 2
# Fix 4: lr_backbone 1e-4, lr_head 5e-4
# Fix 5: keep checkpoint LR when resuming
# Fix 6: prefetch_factor=2 cho DataLoader

import torch
import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "code" else SCRIPT_DIR
CODE_DIR     = os.path.join(PROJECT_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataset import AccidentDataset, get_transforms
from model import get_model, get_optimizer, print_model_info
from utils import collate_fn, compute_map_full, EarlyStopping

IS_COLAB     = os.path.isdir("/content") and "COLAB_GPU" in os.environ
DRIVE_OUTPUT = "/content/drive/MyDrive/HUIT_Project/outputs"
OUTPUT_DIR   = DRIVE_OUTPUT if IS_COLAB and os.path.isdir("/content/drive/MyDrive") \
               else os.path.join(PROJECT_ROOT, "outputs")

# ============================================================
# TRAINING CONFIG
# ============================================================
CONFIG = {
    "split_root"   : os.path.join(PROJECT_ROOT, "Accident.v2i.voc_clean_fixed"),
    "use_predefined_splits": True,
    "data_roots"   : [
        os.path.join(PROJECT_ROOT, "Accident.v2i.voc_clean_fixed", "train"),
        os.path.join(PROJECT_ROOT, "Accident.v2i.voc_clean_fixed", "valid"),
        os.path.join(PROJECT_ROOT, "Accident.v2i.voc_clean_fixed", "test"),
    ],
    "train_split"  : 0.80,
    "val_split"    : 0.10,

    "save_dir"     : os.path.join(OUTPUT_DIR, "weights_clean_fixed"),
    "plot_dir"     : os.path.join(OUTPUT_DIR, "plots_clean_fixed"),
    "history_path" : os.path.join(OUTPUT_DIR, "history_clean_fixed.json"),

    # None = train from scratch | .pth path = resume
    "resume_from"  : None,
    "auto_resume"  : False,

    "backbone"         : "resnet50_v2",
    "num_classes"      : 2,
    "trainable_layers" : 3,
    "img_size"         : 640,

    "num_epochs"   : 30,
    "batch_size"   : 6 if IS_COLAB else 2,
    "num_workers"  : 2,

    # Lower LR to avoid divergence seen in earlier runs
    "lr_backbone"  : 5e-5,
    "lr_head"      : 5e-4,
    "cap_resume_lr": True,
    "momentum"     : 0.9,
    "weight_decay" : 0.0005,

    "lr_scheduler" : True,
    "lr_patience"  : 5,
    "lr_factor"    : 0.5,

    "early_stop"   : True,
    "es_patience"  : 10,

    "save_every"   : 5,
    "plot_every"   : 2,
    "val_every"    : 2 if IS_COLAB else 1,
}

# ============================================================
# HISTORY
# ============================================================
HISTORY_KEYS = ["train_loss", "map_scores", "lr",
                "loss_cls", "loss_box", "loss_obj", "loss_rpn"]

def _empty_history():
    h = {k: [] for k in HISTORY_KEYS}
    h["pr_recall"]    = []
    h["pr_precision"] = []
    return h

def _fix_history(data):
    fixed = _empty_history()
    for k in HISTORY_KEYS:
        fixed[k] = list(data.get(k, []))
    min_len = min(len(fixed[k]) for k in HISTORY_KEYS)
    for k in HISTORY_KEYS:
        fixed[k] = fixed[k][:min_len]
    for key in ["pr_recall", "pr_precision"]:
        arr = data.get(key, [])
        if len(arr) > 100:
            idx        = np.linspace(0, len(arr) - 1, 100).astype(int)
            fixed[key] = [arr[i] for i in idx]
        else:
            fixed[key] = list(arr)
    return fixed, min_len

def load_history(path):
    if not (path and os.path.isfile(path)):
        print("[History] No history.json -> start fresh")
        return _empty_history(), 0
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[Warning] Cannot read history.json ({e}) -> start fresh")
        return _empty_history(), 0
    fixed, n_epochs = _fix_history(raw)
    if n_epochs != len(raw.get("train_loss", [])) or \
       len(raw.get("pr_recall", [])) > 100:
        print("[History] Found invalid history.json -> auto-fix and save")
        save_history(fixed, path)
    print(f"[History] Loaded history: {n_epochs} trained epochs")
    return fixed, n_epochs

def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _source_key(pair):
    """Group Roboflow augmented variants of the same original image."""
    base = os.path.splitext(os.path.basename(pair[0]))[0]
    return base.split(".rf.")[0]


# ============================================================
# PLOTTING
# ============================================================
SNAPSHOT_COLORS = {
    1:  "#e74c3c",
    5:  "#f39c12",
    10: "#27ae60",
    15: "#2980b9",
    20: "#8e44ad",
    25: "#16a085",
    30: "#2c3e50",
    40: "#c0392b",
    50: "#1abc9c",
    60: "#d35400",
    70: "#7f8c8d",
    80: "#2ecc71",
}
SNAPSHOT_EPOCHS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]


def _save_pr_snapshot(history, epoch, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    snap = {
        "epoch"    : epoch,
        "recall"   : list(history["pr_recall"]),
        "precision": list(history["pr_precision"]),
        "map"      : history["map_scores"][-1] if history["map_scores"] else 0.0,
    }
    with open(os.path.join(plot_dir, f"pr_snapshot_epoch_{epoch:03d}.json"), "w") as f:
        json.dump(snap, f)
    return snap


def _load_all_pr_snapshots(plot_dir):
    snaps = {}
    if not os.path.isdir(plot_dir):
        return snaps
    for fname in sorted(os.listdir(plot_dir)):
        if fname.startswith("pr_snapshot_epoch_") and fname.endswith(".json"):
            try:
                with open(os.path.join(plot_dir, fname)) as f:
                    data = json.load(f)
                snaps[data["epoch"]] = data
            except Exception:
                pass
    return snaps


def plot_training(history, plot_dir, epoch):
    os.makedirs(plot_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        return

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(f"Training Dashboard - Epoch {epoch}", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, linewidth=1.5, label="Train loss")
    for se in SNAPSHOT_EPOCHS:
        if se <= len(epochs) and se in SNAPSHOT_COLORS:
            ax1.axvline(se, color=SNAPSHOT_COLORS[se], linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_title("Total Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["map_scores"], "g-o", markersize=4, linewidth=1.5, label="mAP@0.5")
    for se in SNAPSHOT_EPOCHS:
        if se <= len(epochs) and se in SNAPSHOT_COLORS:
            ax2.axvline(se, color=SNAPSHOT_COLORS[se], linestyle="--", alpha=0.5, linewidth=1)
    if history["map_scores"]:
        best_e = int(np.argmax(history["map_scores"])) + 1
        best_v = max(history["map_scores"])
        ax2.annotate(f"Best\n{best_v:.3f}", xy=(best_e, best_v),
                     xytext=(best_e + 0.5, best_v + 0.02),
                     fontsize=8, color="darkgreen",
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1))
    ax2.set_title("mAP@0.5"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP")
    ax2.set_ylim(0, max(1, max(history["map_scores"]) * 1.2) if history["map_scores"] else 1)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for key, color, lbl in [
        ("loss_cls", "#e74c3c", "Classifier"),
        ("loss_box", "#3498db", "Box reg"),
        ("loss_obj", "#2ecc71", "Objectness"),
        ("loss_rpn", "#f39c12", "RPN box"),
    ]:
        if history[key]:
            ax3.plot(epochs, history[key], "-o", markersize=3, linewidth=1.5, color=color, label=lbl)
    ax3.set_title("Loss Breakdown"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history["lr"], "m-o", markersize=4, linewidth=1.5, label="LR")
    ax4.set_title("Learning Rate"); ax4.set_xlabel("Epoch"); ax4.set_ylabel("LR")
    ax4.set_yscale("log"); ax4.legend(); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    if history["pr_recall"] and history["pr_precision"]:
        map_now = history["map_scores"][-1] if history["map_scores"] else 0
        c = SNAPSHOT_COLORS.get(epoch, "#2c3e50")
        ax5.plot(history["pr_recall"], history["pr_precision"], color=c, linewidth=2,
                 label=f"Epoch {epoch}  AP={map_now:.3f}")
        ax5.fill_between(history["pr_recall"], history["pr_precision"], alpha=0.12, color=c)
    ax5.set_title("PR Curve"); ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.set_xlim(0, max(history["pr_recall"] or [1]) * 1.05)
    ax5.set_ylim(0, 1); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    if history["pr_recall"] and history["pr_precision"]:
        rec = np.array(history["pr_recall"])
        pre = np.array(history["pr_precision"])
        f1  = 2 * pre * rec / (pre + rec + 1e-8)
        c   = SNAPSHOT_COLORS.get(epoch, "#2c3e50")
        ax6.plot(rec, f1, "-", linewidth=2, color=c,
                 label=f"Epoch {epoch}  F1={f1.max():.3f}")
        ax6.axvline(rec[f1.argmax()], color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax6.fill_between(rec, f1, alpha=0.12, color=c)
    ax6.set_title("F1 Curve"); ax6.set_xlabel("Recall"); ax6.set_ylabel("F1")
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    fig.savefig(os.path.join(plot_dir, f"training_dashboard_epoch_{epoch:03d}.png"), dpi=120, bbox_inches="tight")
    fig.savefig(os.path.join(plot_dir, "training_dashboard_latest.png"),              dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Dashboard saved - epoch {epoch}")

    if epoch not in SNAPSHOT_EPOCHS:
        return

    _save_pr_snapshot(history, epoch, plot_dir)
    all_snaps = _load_all_pr_snapshots(plot_dir)
    if not all_snaps:
        return

    fig2 = plt.figure(figsize=(20, 13))
    fig2.suptitle(f"Snapshot Compare - Epoch {epoch}", fontsize=14, fontweight="bold")
    gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.35)

    ax_pr = fig2.add_subplot(gs2[0, 0:2])
    for ep_s, snap in sorted(all_snaps.items()):
        c = SNAPSHOT_COLORS.get(ep_s, "#95a5a6")
        if snap["recall"] and snap["precision"]:
            ax_pr.plot(snap["recall"], snap["precision"], color=c, linewidth=2,
                       label=f"Epoch {ep_s}  AP={snap['map']:.3f}")
    ax_pr.set_title("PR Curve - Compare Snapshots")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.legend(fontsize=8, loc="upper right"); ax_pr.grid(True, alpha=0.3)

    ax_f1 = fig2.add_subplot(gs2[0, 2])
    for ep_s, snap in sorted(all_snaps.items()):
        c = SNAPSHOT_COLORS.get(ep_s, "#95a5a6")
        if snap["recall"] and snap["precision"]:
            rec = np.array(snap["recall"])
            pre = np.array(snap["precision"])
            f1  = 2 * pre * rec / (pre + rec + 1e-8)
            ax_f1.plot(rec, f1, color=c, linewidth=2,
                       label=f"Epoch {ep_s}  F1={f1.max():.3f}")
    ax_f1.set_title("F1 Curve - Compare Snapshots")
    ax_f1.set_xlabel("Recall"); ax_f1.set_ylabel("F1")
    ax_f1.set_xlim(0, 1); ax_f1.set_ylim(0, 1)
    ax_f1.legend(fontsize=7, loc="upper right"); ax_f1.grid(True, alpha=0.3)

    ax_loss2 = fig2.add_subplot(gs2[1, 0:2])
    ax_map2  = ax_loss2.twinx()
    ax_loss2.plot(epochs, history["train_loss"], "b-o", markersize=3, linewidth=1.5, label="Loss", alpha=0.7)
    ax_map2.plot(epochs,  history["map_scores"],  "g--s", markersize=3, linewidth=1.5, label="mAP@0.5", alpha=0.7)
    for ep_s in sorted(all_snaps.keys()):
        c = SNAPSHOT_COLORS.get(ep_s, "#95a5a6")
        ax_loss2.axvline(ep_s, color=c, linestyle=":", linewidth=1.8, alpha=0.8)
        ax_loss2.text(ep_s + 0.1, max(history["train_loss"]) * 0.98,
                      f"E{ep_s}", fontsize=7, color=c, va="top")
    ax_loss2.set_xlabel("Epoch")
    ax_loss2.set_ylabel("Loss",    color="blue")
    ax_map2.set_ylabel("mAP@0.5", color="green")
    ax_loss2.set_title("Loss & mAP - snapshot highlights")
    h1, l1 = ax_loss2.get_legend_handles_labels()
    h2, l2 = ax_map2.get_legend_handles_labels()
    ax_loss2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax_loss2.grid(True, alpha=0.3)

    ax_bar    = fig2.add_subplot(gs2[1, 2])
    snap_eps  = sorted(all_snaps.keys())
    snap_maps = [all_snaps[e]["map"] for e in snap_eps]
    snap_cols = [SNAPSHOT_COLORS.get(e, "#95a5a6") for e in snap_eps]
    bars = ax_bar.bar([f"E{e}" for e in snap_eps], snap_maps,
                      color=snap_cols, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, snap_maps):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_title("mAP at snapshot epochs")
    ax_bar.set_xlabel("Epoch"); ax_bar.set_ylabel("mAP@0.5")
    ax_bar.set_ylim(0, max(snap_maps) * 1.25 if snap_maps else 1)
    ax_bar.grid(True, alpha=0.3, axis="y")

    fig2.savefig(os.path.join(plot_dir, f"snapshot_compare_epoch_{epoch:03d}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Snapshot comparison saved - epoch {epoch}")


def plot_confusion_matrix(cm, class_names, plot_dir, epoch):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Confusion Matrix - Epoch {epoch}", fontweight="bold")
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)],
        ["Raw counts", "Normalized"], ["d", ".2f"]
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color="white" if val > data.max() * 0.6 else "black", fontsize=9)
    fig.savefig(os.path.join(plot_dir, f"confusion_matrix_epoch_{epoch:03d}.png"), dpi=120, bbox_inches="tight")
    fig.savefig(os.path.join(plot_dir, "confusion_matrix_latest.png"),              dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Confusion matrix saved - epoch {epoch}")


# ============================================================
# EVALUATE TEST SET
# ============================================================
@torch.no_grad()
def evaluate_test(model, data_loader, device, num_classes, class_names, plot_dir):
    model.eval()
    predictions, ground_truths = [], []
    print("\n" + "="*60)
    print("  FINAL EVALUATION ON TEST SET")
    print("="*60)
    for images, targets in tqdm(data_loader, desc="Test", unit="img", ncols=100, colour="yellow"):
        images  = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            predictions.append({"boxes": output["boxes"].cpu(), "scores": output["scores"].cpu(), "labels": output["labels"].cpu()})
            ground_truths.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})

    test_map, pr_data, cm = compute_map_full(predictions, ground_truths, num_classes=num_classes)
    print(f"\nTest mAP@0.5 (final): {test_map:.4f}")
    os.makedirs(plot_dir, exist_ok=True)

    if cm is not None:
        plot_confusion_matrix(cm, class_names, plot_dir, epoch="test")

    if pr_data["recall"] and pr_data["precision"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Test Set - mAP@0.5 = {test_map:.4f}", fontsize=13, fontweight="bold")
        axes[0].plot(pr_data["recall"], pr_data["precision"], color="#e74c3c", linewidth=2.5, label=f"AP={test_map:.3f}")
        axes[0].fill_between(pr_data["recall"], pr_data["precision"], alpha=0.15, color="#e74c3c")
        axes[0].set_title("PR Curve - Test Set"); axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
        axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        rec = np.array(pr_data["recall"]); pre = np.array(pr_data["precision"])
        f1  = 2 * pre * rec / (pre + rec + 1e-8)
        axes[1].plot(rec, f1, color="#2980b9", linewidth=2.5, label=f"F1 max={f1.max():.3f}")
        axes[1].fill_between(rec, f1, alpha=0.15, color="#2980b9")
        axes[1].set_title("F1 Curve - Test Set"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("F1")
        axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, "test_set_result.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("[Plot] Test result chart saved")
    return test_map


# ============================================================
# TRAIN 1 EPOCH
# ============================================================
def train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, num_epochs, amp_enabled=True):
    model.train()
    total_loss     = 0
    loss_breakdown = {"loss_classifier": 0, "loss_box_reg": 0, "loss_objectness": 0, "loss_rpn_box_reg": 0}
    pbar  = tqdm(data_loader, desc=f"Train {epoch}/{num_epochs}", unit="batch", ncols=100, colour="blue")
    start = time.time()

    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        for k in loss_breakdown:
            if k in loss_dict:
                loss_breakdown[k] += loss_dict[k].item()
        pbar.set_postfix(loss=f"{losses.item():.4f}", refresh=False)

    elapsed  = time.time() - start
    avg_loss = total_loss / len(data_loader)
    n        = len(data_loader)
    print(f"\nEpoch [{epoch}] - Avg Loss: {avg_loss:.4f} | {elapsed:.0f}s")
    print(f"   cls={loss_breakdown['loss_classifier']/n:.3f} | "
          f"box={loss_breakdown['loss_box_reg']/n:.3f} | "
          f"obj={loss_breakdown['loss_objectness']/n:.3f} | "
          f"rpn={loss_breakdown['loss_rpn_box_reg']/n:.3f}")
    return avg_loss, {k: v/n for k, v in loss_breakdown.items()}


# ============================================================
# VALIDATE
# ============================================================
@torch.no_grad()
def validate(model, data_loader, device, num_classes, epoch, num_epochs):
    model.eval()
    predictions, ground_truths = [], []
    for images, targets in tqdm(data_loader, desc=f"Val {epoch}/{num_epochs}", unit="img", ncols=100, colour="green"):
        images  = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            predictions.append({"boxes": output["boxes"].cpu(), "scores": output["scores"].cpu(), "labels": output["labels"].cpu()})
            ground_truths.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})
    map_score, pr_data, cm = compute_map_full(predictions, ground_truths, num_classes=num_classes)
    print(f"mAP@0.5: {map_score:.4f}")
    return map_score, pr_data, cm


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CONFIG
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["plot_dir"], exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"   weights : {cfg['save_dir']}")
    print(f"   plots   : {cfg['plot_dir']}")
    print(f"   history : {cfg['history_path']}")

    for root in cfg["data_roots"]:
        if not os.path.exists(root):
            print(f"Missing dataset split: {root}"); return

    def make_dataset(pairs, train_aug):
        ds            = AccidentDataset.__new__(AccidentDataset)
        ds.pairs      = pairs
        ds.transforms = get_transforms(train=train_aug, img_size=cfg["img_size"])
        return ds

    if cfg.get("use_predefined_splits", False):
        print("\nUsing predefined train/valid/test folders:")
        train_dataset = AccidentDataset(cfg["data_roots"][0], transforms=get_transforms(True, cfg["img_size"]), skip_empty=False)
        val_dataset   = AccidentDataset(cfg["data_roots"][1], transforms=get_transforms(False, cfg["img_size"]), skip_empty=False)
        test_dataset  = AccidentDataset(cfg["data_roots"][2], transforms=get_transforms(False, cfg["img_size"]), skip_empty=False)

        n_train = len(train_dataset)
        n_val   = len(val_dataset)
        n_test  = len(test_dataset)
        n_total = n_train + n_val + n_test

        split_sources = []
        for dataset in (train_dataset, val_dataset, test_dataset):
            split_sources.append({_source_key(pair) for pair in dataset.pairs})
        overlap_train_val = len(split_sources[0] & split_sources[1])
        overlap_train_test = len(split_sources[0] & split_sources[2])
        overlap_val_test = len(split_sources[1] & split_sources[2])

        print(f"   Total : {n_total:>6} images")
        print(f"   Train : {n_train:>6} images ({n_train / max(n_total, 1) * 100:5.2f}%)")
        print(f"   Val   : {n_val:>6} images ({n_val / max(n_total, 1) * 100:5.2f}%)")
        print(f"   Test  : {n_test:>6} images ({n_test / max(n_total, 1) * 100:5.2f}%)")
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("WARNING: source-image overlap between splits:")
            print(f"   train/val={overlap_train_val}, train/test={overlap_train_test}, val/test={overlap_val_test}")
    else:
        print("\nChecking data_roots and creating grouped 80/10/10 split:")
        all_pairs  = []
        seen_paths = set()
        for root in cfg["data_roots"]:
            tmp    = AccidentDataset(root, transforms=None, skip_empty=True)
            before = len(all_pairs)
            for pair in tmp.pairs:
                if pair[0] not in seen_paths:
                    seen_paths.add(pair[0])
                    all_pairs.append(pair)
            added = len(all_pairs) - before
            dup   = len(tmp.pairs) - added
            print(f"   {root}")
            print(f"   -> {len(tmp.pairs)} images | added {added} | skipped {dup} duplicates")

        n_total = len(all_pairs)
        groups = {}
        for pair in all_pairs:
            groups.setdefault(_source_key(pair), []).append(pair)

        rng = np.random.default_rng(42)
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        target_train = max(1, int(n_total * cfg["train_split"]))
        target_val   = max(1, int(n_total * cfg["val_split"]))

        train_pairs, val_pairs, test_pairs = [], [], []
        train_groups, val_groups, test_groups = 0, 0, 0
        for key in group_keys:
            bucket = groups[key]
            if len(train_pairs) < target_train:
                train_pairs.extend(bucket)
                train_groups += 1
            elif len(val_pairs) < target_val:
                val_pairs.extend(bucket)
                val_groups += 1
            else:
                test_pairs.extend(bucket)
                test_groups += 1

        n_train = len(train_pairs)
        n_val   = len(val_pairs)
        n_test  = len(test_pairs)

        train_dataset = make_dataset(train_pairs, True)
        val_dataset   = make_dataset(val_pairs,   False)
        test_dataset  = make_dataset(test_pairs,  False)

        print(f"\nSplit 80/10/10 (grouped by source image, seed=42):")
        print(f"   Total : {n_total:>6} images")
        print(f"   Train : {n_train:>6} images (80%)")
        print(f"   Val   : {n_val:>6} images (10%)")
        print(f"   Test  : {n_test:>6} images (10%)")
        print(f"   Groups: {len(groups)} source images | train={train_groups}, val={val_groups}, test={test_groups}")

    pin = (device.type == "cuda")
    pw  = (cfg["num_workers"] > 0)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], collate_fn=collate_fn,
                              pin_memory=pin, persistent_workers=pw, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_fn,
                              pin_memory=pin, persistent_workers=pw, prefetch_factor=2)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False,
                              num_workers=cfg["num_workers"], collate_fn=collate_fn,
                              pin_memory=pin, persistent_workers=pw, prefetch_factor=2)


    model     = get_model(cfg["num_classes"], cfg["backbone"], cfg["trainable_layers"])
    model.to(device)
    print_model_info(model)

    optimizer = get_optimizer(model,
                              lr_backbone  = cfg["lr_backbone"],
                              lr_head      = cfg["lr_head"],
                              momentum     = cfg["momentum"],
                              weight_decay = cfg["weight_decay"])

    last_path = os.path.join(cfg["save_dir"], "last_model.pth")
    if cfg.get("resume_from") is None and cfg.get("auto_resume") and os.path.isfile(last_path):
        cfg["resume_from"] = last_path
        print(f"Auto resume: {last_path}")

    can_resume = bool(cfg.get("resume_from") and os.path.isfile(cfg["resume_from"]))
    if can_resume:
        history, history_epochs = load_history(cfg.get("history_path"))
    else:
        history, history_epochs = _empty_history(), 0
    start_epoch = 1
    best_map    = 0.0

    if can_resume:
        print(f"\nLoad checkpoint: {cfg['resume_from']}")
        ckpt = torch.load(cfg["resume_from"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "map"   in ckpt:
            best_map    = ckpt["map"]

        if history_epochs == 0 and "history" in ckpt:
            ckpt_history, ckpt_epochs = _fix_history(ckpt["history"])
            history        = ckpt_history
            history_epochs = ckpt_epochs
            print(f"   Use history from checkpoint ({ckpt_epochs} epochs)")

        expected = start_epoch - 1
        if history_epochs > expected:
            for k in HISTORY_KEYS:
                history[k] = history[k][:expected]
            print(f"   Trim history to {expected} epochs")

        if cfg.get("cap_resume_lr", True):
            max_lrs = [cfg["lr_backbone"], cfg["lr_head"]]
            for pg, max_lr in zip(optimizer.param_groups, max_lrs):
                if pg["lr"] > max_lr:
                    pg["lr"] = max_lr

        # Keep checkpoint LR, only cap old checkpoints above the safe LR.
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]
        print(f"   Resume from epoch {start_epoch} | Best mAP: {best_map:.4f}")
        print(f"   Keep checkpoint LR: backbone={lr_bb:.2e} | head={lr_hd:.2e}")
    else:
        print("\nTrain from scratch")
        # Set LR only when training from scratch.
        optimizer.param_groups[0]["lr"] = cfg["lr_backbone"]
        optimizer.param_groups[1]["lr"] = cfg["lr_head"]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["lr_factor"], patience=cfg["lr_patience"]
    ) if cfg["lr_scheduler"] else None

    early_stop = EarlyStopping(patience=cfg["es_patience"], mode="max") \
                 if cfg["early_stop"] else None

    best_path   = os.path.join(cfg["save_dir"], "best_model.pth")
    class_names = ["background", "accident"]

    print(f"\nTraining from epoch {start_epoch} -> {cfg['num_epochs']}\n")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch}/{cfg['num_epochs']}   LR backbone: {lr_bb:.2e} | LR head: {lr_hd:.2e}")
        print(f"{'='*60}")

        train_loss, breakdown = train_one_epoch(
            model, optimizer, scaler, train_loader, device, epoch, cfg["num_epochs"], amp_enabled=amp_enabled)

        should_validate = (epoch == start_epoch) or (epoch % cfg["val_every"] == 0)
        if should_validate:
            map_score, pr_data, cm = validate(
                model, val_loader, device, cfg["num_classes"], epoch, cfg["num_epochs"])
        else:
            map_score = history["map_scores"][-1] if history["map_scores"] else best_map
            pr_data = {
                "recall": history.get("pr_recall", []),
                "precision": history.get("pr_precision", []),
            }
            cm = None
            print(f"Skip validation epoch {epoch} to save Colab time")

        history["train_loss"].append(train_loss)
        history["map_scores"].append(map_score)
        history["lr"].append(lr_hd)
        history["loss_cls"].append(breakdown["loss_classifier"])
        history["loss_box"].append(breakdown["loss_box_reg"])
        history["loss_obj"].append(breakdown["loss_objectness"])
        history["loss_rpn"].append(breakdown["loss_rpn_box_reg"])
        history["pr_recall"]    = pr_data["recall"]
        history["pr_precision"] = pr_data["precision"]

        save_history(history, cfg["history_path"])

        if epoch % cfg["plot_every"] == 0:
            plot_training(history, cfg["plot_dir"], epoch)
            if cm is not None:
                plot_confusion_matrix(cm, class_names, cfg["plot_dir"], epoch)

        if scheduler is not None and should_validate:
            scheduler.step(map_score)

        if should_validate and map_score > best_map:
            best_map = map_score
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "map": best_map,
                "config": cfg, "history": history,
            }, best_path)
            print(f"Best mAP={best_map:.4f} -> {best_path}")

        if epoch % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["save_dir"], f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "map": map_score,
                "config": cfg, "history": history,
            }, ckpt_path)
            print(f"Checkpoint: {ckpt_path}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(), "map": map_score,
            "history": history,
        }, os.path.join(cfg["save_dir"], "last_model.pth"))

        if should_validate and early_stop is not None and early_stop.step(map_score):
            print(f"\nEarly stopping at epoch {epoch}. Best mAP: {best_map:.4f}")
            break

    final_path = os.path.join(cfg["save_dir"], "final_model.pth")
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(), "map": best_map,
        "config": cfg, "history": history,
    }, final_path)
    plot_training(history, cfg["plot_dir"], epoch)

    print(f"\nLoad best_model for test evaluation: {best_path}")
    if os.path.isfile(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
    else:
        print("[Warning] best_model.pth not found -> using current weights")
        best_ckpt = None

    test_map = evaluate_test(model, test_loader, device,
                             num_classes=cfg["num_classes"],
                             class_names=class_names,
                             plot_dir=cfg["plot_dir"])

    if best_ckpt is not None and os.path.isfile(best_path):
        best_ckpt["test_map"] = test_map
        torch.save(best_ckpt, best_path)
        print(f"Saved test_map={test_map:.4f} into best_model.pth")

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"  Val  mAP@0.5  (best): {best_map:.4f}")
    print(f"  Test mAP@0.5 (final): {test_map:.4f}")
    print(f"{'='*60}")
    print(f"   Best   : {best_path}")
    print(f"   Final  : {final_path}")
    print(f"   History: {cfg['history_path']}")
    print(f"   Charts : {cfg['plot_dir']}")


if __name__ == "__main__":
    main()




