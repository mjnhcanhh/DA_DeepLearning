import os
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from tqdm import tqdm

import config
from dataset import VOCDataset, collate_fn


def create_model(num_classes: int):
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

    in_channels = []
    for block in model.head.regression_head.module_list:
        for layer in block:
            if isinstance(layer, torch.nn.Conv2d):
                in_channels.append(layer.in_channels)
                break

    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.BatchNorm2d
    )
    return model


def save_checkpoint(model, optimizer, epoch, best_val_loss, class_names, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "class_names": class_names
    }, save_path)


def load_checkpoint(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    return start_epoch, best_val_loss


def append_history_csv(epoch, train_loss, val_loss, is_best):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    csv_path = os.path.join(config.LOG_DIR, "train_history.csv")

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "is_best", "timestamp"])
        writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            "YES" if is_best else "",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Đang dùng thiết bị:", device)

    train_dataset = VOCDataset(
        images_dir=config.TRAIN_IMAGES,
        annotations_dir=config.TRAIN_ANNOS,
        class_names=config.CLASS_NAMES,
        is_train=True,
        image_size=config.IMAGE_SIZE
    )

    val_dataset = VOCDataset(
        images_dir=config.VAL_IMAGES,
        annotations_dir=config.VAL_ANNOS,
        class_names=config.CLASS_NAMES,
        is_train=False,
        image_size=config.IMAGE_SIZE
    )

    print("Số ảnh train:", len(train_dataset))
    print("Số ảnh val:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = create_model(config.NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(config.LAST_CHECKPOINT_PATH):
        print("Tìm thấy checkpoint cũ, tiếp tục train...")
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, config.LAST_CHECKPOINT_PATH, device
        )
        print(f"Resume từ epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN]")
        for images, targets in train_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_train_loss += losses.item()
            train_steps += 1
            train_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_train_loss = total_train_loss / max(train_steps, 1)

        model.train()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [VAL]")
            for images, targets in val_bar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_val_loss += losses.item()
                val_steps += 1
                val_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_val_loss = total_val_loss / max(val_steps, 1)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                config.CLASS_NAMES, config.MODEL_SAVE_PATH
            )
            print("Đã lưu best model:", config.MODEL_SAVE_PATH)

        save_checkpoint(
            model, optimizer, epoch, best_val_loss,
            config.CLASS_NAMES, config.LAST_CHECKPOINT_PATH
        )

        append_history_csv(epoch, avg_train_loss, avg_val_loss, is_best)

        print(f"\nEpoch {epoch+1}:")
        print(f"  train_loss = {avg_train_loss:.4f}")
        print(f"  val_loss   = {avg_val_loss:.4f}")

    print("\nHuấn luyện xong.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()