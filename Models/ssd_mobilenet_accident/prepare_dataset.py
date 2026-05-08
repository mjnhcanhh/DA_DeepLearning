import os
import shutil
import random
import xml.etree.ElementTree as ET
import cv2

# ==============================
# ĐƯỜNG DẪN
# ==============================
SOURCE_DIR = r"D:\DeepLN\DA_DeepLearning\DA_DeepLearning\Data\Accident.v1i.voc"
TARGET_DIR = r"D:\DeepLN\DA_DeepLearning\DA_DeepLearning\Data\ssd_dataset"

IMAGE_SIZE = 320
RANDOM_SEED = 42

# Nếu dataset đã có sẵn train/valid/test thì để True
USE_EXISTING_SPLITS = True


def make_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TARGET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, "annotations", split), exist_ok=True)


def resize_and_save(image_path, xml_path, out_img_path, out_xml_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return False

    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # cập nhật size trong xml nếu có
    size = root.find("size")
    if size is not None:
        width_tag = size.find("width")
        height_tag = size.find("height")
        depth_tag = size.find("depth")

        if width_tag is not None:
            width_tag.text = str(IMAGE_SIZE)
        if height_tag is not None:
            height_tag.text = str(IMAGE_SIZE)
        if depth_tag is not None:
            depth_tag.text = str(3)

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin_tag = bndbox.find("xmin")
        ymin_tag = bndbox.find("ymin")
        xmax_tag = bndbox.find("xmax")
        ymax_tag = bndbox.find("ymax")

        if None in [xmin_tag, ymin_tag, xmax_tag, ymax_tag]:
            continue

        xmin = int(float(xmin_tag.text))
        ymin = int(float(ymin_tag.text))
        xmax = int(float(xmax_tag.text))
        ymax = int(float(ymax_tag.text))

        xmin = max(0, int(xmin * IMAGE_SIZE / w))
        ymin = max(0, int(ymin * IMAGE_SIZE / h))
        xmax = min(IMAGE_SIZE - 1, int(xmax * IMAGE_SIZE / w))
        ymax = min(IMAGE_SIZE - 1, int(ymax * IMAGE_SIZE / h))

        xmin_tag.text = str(xmin)
        ymin_tag.text = str(ymin)
        xmax_tag.text = str(xmax)
        ymax_tag.text = str(ymax)

    cv2.imwrite(out_img_path, img_resized)
    tree.write(out_xml_path, encoding="utf-8")
    return True


def process_existing_split(src_split_name, dst_split_name):
    src_dir = os.path.join(SOURCE_DIR, src_split_name)
    if not os.path.exists(src_dir):
        print(f"Không thấy thư mục: {src_dir}")
        return

    all_files = os.listdir(src_dir)
    image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"\n=== Xử lí split {src_split_name} -> {dst_split_name} ===")
    print("Số ảnh tìm thấy:", len(image_files))

    ok_count = 0
    miss_xml = 0

    for image_file in image_files:
        base = os.path.splitext(image_file)[0]
        xml_file = base + ".xml"

        image_path = os.path.join(src_dir, image_file)
        xml_path = os.path.join(src_dir, xml_file)

        if not os.path.exists(xml_path):
            miss_xml += 1
            print("Thiếu xml:", image_file)
            continue

        out_img_path = os.path.join(TARGET_DIR, "images", dst_split_name, image_file)
        out_xml_path = os.path.join(TARGET_DIR, "annotations", dst_split_name, xml_file)

        if resize_and_save(image_path, xml_path, out_img_path, out_xml_path):
            ok_count += 1

    print(f"Hoàn tất {dst_split_name}: {ok_count} ảnh/xml")
    print(f"Thiếu xml: {miss_xml}")


def process_and_split_from_one_folder():
    all_files = os.listdir(SOURCE_DIR)
    image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print("Tổng số ảnh:", len(image_files))

    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    n = len(image_files)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    split_map = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    for split, files in split_map.items():
        print(f"\n=== Xử lí split {split} ===")
        ok_count = 0
        miss_xml = 0

        for image_file in files:
            base = os.path.splitext(image_file)[0]
            xml_file = base + ".xml"

            image_path = os.path.join(SOURCE_DIR, image_file)
            xml_path = os.path.join(SOURCE_DIR, xml_file)

            if not os.path.exists(xml_path):
                miss_xml += 1
                print("Thiếu xml:", image_file)
                continue

            out_img_path = os.path.join(TARGET_DIR, "images", split, image_file)
            out_xml_path = os.path.join(TARGET_DIR, "annotations", split, xml_file)

            if resize_and_save(image_path, xml_path, out_img_path, out_xml_path):
                ok_count += 1

        print(f"Hoàn tất {split}: {ok_count} ảnh/xml")
        print(f"Thiếu xml: {miss_xml}")


def main():
    make_dirs()

    if USE_EXISTING_SPLITS:
        process_existing_split("train", "train")
        process_existing_split("valid", "val")
        process_existing_split("test", "test")
    else:
        process_and_split_from_one_folder()

    print("\n✅ Hoàn tất chuẩn bị dataset cho SSD")


if __name__ == "__main__":
    main()