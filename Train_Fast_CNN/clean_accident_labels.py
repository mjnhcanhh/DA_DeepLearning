import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
TARGET_NAMES = {"accident"}


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def get_size(root):
    size = root.find("size")
    if size is None:
        return 0, 0
    width = int(float(size.findtext("width", "0") or 0))
    height = int(float(size.findtext("height", "0") or 0))
    return width, height


def read_box(obj):
    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None
    vals = []
    for name in ("xmin", "ymin", "xmax", "ymax"):
        text = bndbox.findtext(name)
        if text is None:
            return None
        vals.append(float(text))
    xmin, ymin, xmax, ymax = vals
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return xmin, ymin, xmax, ymax


def write_box(obj, box):
    xmin, ymin, xmax, ymax = box
    bndbox = obj.find("bndbox")
    for name, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
        node = bndbox.find(name)
        if node is None:
            node = ET.SubElement(bndbox, name)
        node.text = str(int(round(value)))


def clean_xml(src_xml, dst_xml, args):
    tree = ET.parse(src_xml)
    root = tree.getroot()
    width, height = get_size(root)
    image_area = max(width * height, 1)

    kept = []
    stats = {
        "objects_total": 0,
        "kept_accident": 0,
        "removed_non_accident": 0,
        "removed_invalid": 0,
        "removed_too_small": 0,
        "removed_too_large": 0,
        "clipped": 0,
    }

    for obj in list(root.findall("object")):
        stats["objects_total"] += 1
        raw_name = (obj.findtext("name") or "").strip()
        if raw_name.lower() not in TARGET_NAMES:
            stats["removed_non_accident"] += 1
            continue

        box = read_box(obj)
        if box is None or width <= 0 or height <= 0:
            stats["removed_invalid"] += 1
            continue

        xmin, ymin, xmax, ymax = box
        clipped = (
            max(0.0, min(xmin, width)),
            max(0.0, min(ymin, height)),
            max(0.0, min(xmax, width)),
            max(0.0, min(ymax, height)),
        )
        if clipped != box:
            stats["clipped"] += 1
        xmin, ymin, xmax, ymax = clipped
        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w < args.min_side or box_h < args.min_side or box_w * box_h < args.min_area:
            stats["removed_too_small"] += 1
            continue

        area_ratio = (box_w * box_h) / image_area
        if area_ratio > args.max_area_ratio:
            stats["removed_too_large"] += 1
            continue

        cleaned_obj = deepcopy(obj)
        cleaned_obj.find("name").text = "Accident"
        write_box(cleaned_obj, (xmin, ymin, xmax, ymax))
        kept.append(cleaned_obj)
        stats["kept_accident"] += 1

    for obj in list(root.findall("object")):
        root.remove(obj)
    for obj in kept:
        root.append(obj)

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=False)
    return stats


def link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Clean Pascal VOC Accident labels for training.")
    parser.add_argument("--src", default="Accident.v2i.voc_80_10_10")
    parser.add_argument("--dst", default="Accident.v2i.voc_clean_fixed")
    parser.add_argument("--max-area-ratio", type=float, default=1.00)
    parser.add_argument("--min-side", type=float, default=4.0)
    parser.add_argument("--min-area", type=float, default=16.0)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        raise SystemExit(f"Source dataset not found: {src_root}")
    if dst_root.exists() and args.overwrite:
        shutil.rmtree(dst_root)
    if dst_root.exists():
        existing_files = [p for p in dst_root.rglob("*") if p.is_file()]
        if existing_files:
            raise SystemExit(f"Destination already exists and contains files: {dst_root}")

    report = {
        "source": str(src_root),
        "destination": str(dst_root),
        "max_area_ratio": args.max_area_ratio,
        "min_side": args.min_side,
        "min_area": args.min_area,
        "splits": {},
        "totals": {},
    }
    totals = {}

    for split in ("train", "valid", "test"):
        src_split = src_root / split
        dst_split = dst_root / split
        split_stats = {"images": 0, "xml": 0, "empty_after_clean": 0}
        for key in (
            "objects_total", "kept_accident", "removed_non_accident", "removed_invalid",
            "removed_too_small", "removed_too_large", "clipped"
        ):
            split_stats[key] = 0

        xmls = {p.stem: p for p in src_split.glob("*.xml")}
        for img in src_split.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            xml = xmls.get(img.stem)
            if xml is None:
                continue
            dst_img = dst_split / img.name
            dst_xml = dst_split / xml.name
            link_or_copy(img, dst_img, args.mode)
            stats = clean_xml(xml, dst_xml, args)
            split_stats["images"] += 1
            split_stats["xml"] += 1
            if stats["kept_accident"] == 0:
                split_stats["empty_after_clean"] += 1
            for key, value in stats.items():
                split_stats[key] += value

        report["splits"][split] = split_stats
        for key, value in split_stats.items():
            totals[key] = totals.get(key, 0) + value

    report["totals"] = totals
    dst_root.mkdir(parents=True, exist_ok=True)
    report_path = dst_root / "clean_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Created clean dataset: {dst_root}")
    print(f"Report: {report_path}")
    print(json.dumps(report["splits"], indent=2))
    print("TOTALS")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()


