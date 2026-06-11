# dataset.py - SUA BUG TRANSFORM KHONG SYNC BOX
# Dung albumentations thay torchvision transforms
#  flip/rotate/jitter eu uoc apply ung len ca anh lan bounding box

import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ============================================================
# INPUT SCALING
# ============================================================
# Torchvision detection models already normalize images internally.
# The dataset must return float tensors in [0, 1].


def get_transforms(train=True, img_size=640):
    """
    BO TIEN XU LY & TANG CUONG DU LIEU dung albumentations.

    Tat ca augmentation (flip, rotate, jitter, blur) eu uoc
    sync tu ong voi bounding box thong qua BboxParams.

    Train pipeline:
        Resize  HorizontalFlip  ShiftScaleRotate
         ColorJitter  GaussianBlur  ToGray (nhe)
         ToFloat([0,1])  ToTensorV2

    Val/Test pipeline:
        Resize  ToFloat([0,1])  ToTensorV2
    """
    bbox_params = A.BboxParams(
        format="pascal_voc",          # [xmin, ymin, xmax, ymax]  khop voi VOC XML
        label_fields=["labels"],       # sync labels tuong ung voi box
        min_area=4,                    # bo box qua nho sau transform (< 4 pixel2)
        min_visibility=0.2,            # bo box bi crop/rotate mat > 80% dien tich
        clip=True,                     # clip box ra ngoai bien anh ve ung bien
    )

    if train:
        transform = A.Compose([
            # 1. Resize co inh
            A.Resize(img_size, img_size),

            # 2. Lat ngang (50%)  box uoc flip theo
            A.HorizontalFlip(p=0.5),

            # 3. Shift + Scale + Rotate nhe  an toan hon RandomRotation thuan
            #    shift_limit=0.05: dich toi a 5%
            #    scale_limit=0.1 : zoom in/out toi a 10%
            #    rotate_limit=10 : xoay 10
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,         # fill vien bang 0 (en)
                p=0.4
            ),

            # 4. Thay oi mau sac
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
                p=0.5
            ),

            # 5. Gaussian blur nhe
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),

            # 6. Random grayscale (10%)  simulate camera en trang/CCTV cu
            A.ToGray(p=0.1),

            # 7. Scale ve [0, 1]. Faster R-CNN se tu normalize ben trong.
            A.ToFloat(max_value=255.0),

            # 8. Chuyen sang tensor PyTorch (H,W,C  C,H,W)
            ToTensorV2(),
        ], bbox_params=bbox_params)

    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    return transform


# ============================================================
# LABEL MAP
# ============================================================
LABEL_MAP = {
    "Accident": 1,
    "accident": 1,
}


class AccidentDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, skip_empty=True):
        self.root       = root
        self.transforms = transforms

        all_imgs = {os.path.splitext(f)[0]: f
                    for f in os.listdir(root) if f.lower().endswith(IMAGE_EXTS)}
        all_xmls = {os.path.splitext(f)[0]: f
                    for f in os.listdir(root) if f.lower().endswith(".xml")}

        common_keys = sorted(set(all_imgs.keys()) & set(all_xmls.keys()))

        self.pairs = []
        for key in common_keys:
            img_path = os.path.join(root, all_imgs[key])
            xml_path = os.path.join(root, all_xmls[key])

            if skip_empty:
                tree = ET.parse(xml_path)
                if len(tree.getroot().findall("object")) == 0:
                    continue

            self.pairs.append((img_path, xml_path))

        print(f"[Dataset] Tim thay {len(self.pairs)} anh hop le tai: {root}")

    def __getitem__(self, idx):
        img_path, xml_path = self.pairs[idx]

        # --- oc anh  numpy uint8 (H, W, C) vi albumentations yeu cau ---
        img = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]

        # --- Parse XML ---
        tree     = ET.parse(xml_path)
        xml_root = tree.getroot()

        boxes, labels = [], []
        for obj in xml_root.findall("object"):
            class_name = obj.find("name").text.strip()
            label      = LABEL_MAP.get(class_name)
            if label is None:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # am bao xmin < xmax, ymin < ymax
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)

            # Clip ve bien anh goc
            xmin = max(0.0, min(xmin, orig_w))
            xmax = max(0.0, min(xmax, orig_w))
            ymin = max(0.0, min(ymin, orig_h))
            ymax = max(0.0, min(ymax, orig_h))

            # Bo box qua nho
            if xmax - xmin > 1 and ymax - ymin > 1:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # --- Ap dung transform (albumentations sync box tu ong) ---
        if self.transforms is not None and len(boxes) > 0:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels,
            )
            img    = transformed["image"]              # tensor (C, H, W)
            boxes  = list(transformed["bboxes"])       # list of (xmin,ymin,xmax,ymax)
            labels = list(transformed["labels"])

        elif self.transforms is not None:
            # Khong co box  chi transform anh
            transformed = self.transforms(
                image=img, bboxes=[], labels=[]
            )
            img = transformed["image"]

        # --- Tao target ---
        if len(boxes) > 0:
            target_boxes = torch.as_tensor(boxes,  dtype=torch.float32)
            target_lbls  = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_lbls  = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes"   : target_boxes,
            "labels"  : target_lbls,
            "image_id": torch.tensor([idx]),
        }

        return img, target

    def __len__(self):
        return len(self.pairs)


