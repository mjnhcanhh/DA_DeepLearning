import os
import random
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T


class VOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, class_names, is_train=False, image_size=320):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.class_names = class_names
        self.is_train = is_train
        self.image_size = image_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.class_names:
                continue

            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_names.index(class_name))

        return boxes, labels

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)

        xml_name = os.path.splitext(image_name)[0] + ".xml"
        xml_path = os.path.join(self.annotations_dir, xml_name)

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = self.parse_xml(xml_path)

        new_w, new_h = self.image_size, self.image_size
        image = image.resize((new_w, new_h), Image.BILINEAR)

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        boxes_scaled = []
        labels_scaled = []

        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            xmin = max(0, int(xmin * scale_x))
            ymin = max(0, int(ymin * scale_y))
            xmax = min(new_w - 1, int(xmax * scale_x))
            ymax = min(new_h - 1, int(ymax * scale_y))

            if xmax > xmin and ymax > ymin:
                boxes_scaled.append([xmin, ymin, xmax, ymax])
                labels_scaled.append(label)

        boxes = boxes_scaled
        labels = labels_scaled

        if self.is_train and len(boxes) > 0:
            if random.random() < 0.5:
                image = F.hflip(image)
                boxes = [[new_w - xmax, ymin, new_w - xmin, ymax] for xmin, ymin, xmax, ymax in boxes]

            image = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )(image)

        image = F.to_tensor(image)
        image = F.normalize(image, self.mean, self.std)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx])
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))