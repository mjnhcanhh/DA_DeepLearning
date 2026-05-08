import os
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

import config


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


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {config.MODEL_SAVE_PATH}")

    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = create_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(image_path, image_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    image_resized = image_pil.resize((image_size, image_size))
    image_tensor = F.to_tensor(image_resized)
    image_tensor = F.normalize(image_tensor, mean, std)

    return image_pil, image_tensor, orig_w, orig_h


def draw_predictions_on_original(image_cv, outputs, class_names, threshold, resized_size, orig_w, orig_h):
    scale_x = orig_w / resized_size
    scale_y = orig_h / resized_size

    boxes = outputs["boxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]

    for box, label, score in zip(boxes, labels, scores):
        score = float(score.item())
        if score < threshold:
            continue

        x1, y1, x2, y2 = box.tolist()

        # scale ngược box từ ảnh resize 320x320 về ảnh gốc
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        class_id = int(label.item())
        if class_id < 0 or class_id >= len(class_names):
            continue

        class_name = class_names[class_id]

        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_cv,
            f"{class_name}: {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return image_cv


def predict_image(image_path: str, save_path: str = None, show_image: bool = True):
    model, class_names, device = load_model()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    _, image_tensor, orig_w, orig_h = preprocess_image(image_path, config.IMAGE_SIZE)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    image_cv = cv2.imread(image_path)
    image_cv = draw_predictions_on_original(
        image_cv=image_cv,
        outputs=outputs,
        class_names=class_names,
        threshold=config.CONFIDENCE_THRESHOLD,
        resized_size=config.IMAGE_SIZE,
        orig_w=orig_w,
        orig_h=orig_h
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_cv)
        print("Đã lưu ảnh kết quả:", save_path)

    if show_image:
        cv2.imshow("SSD Detection", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def predict_first_test_image():
    if not os.path.exists(config.TEST_IMAGES):
        raise FileNotFoundError(f"Không tìm thấy thư mục test images: {config.TEST_IMAGES}")

    image_files = [
        f for f in os.listdir(config.TEST_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise FileNotFoundError("Không có ảnh nào trong thư mục test.")

    image_path = os.path.join(config.TEST_IMAGES, image_files[0])
    print("Đang test ảnh:", image_path)
    predict_image(image_path)


if __name__ == "__main__":
    predict_first_test_image()