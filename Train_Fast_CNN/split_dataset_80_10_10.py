import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def source_key(path: Path) -> str:
    """Keep Roboflow augmented variants of the same source image together."""
    return path.stem.split(".rf.")[0]


def collect_pairs(src_root: Path):
    pairs = []
    seen = set()
    for split in ("train", "valid", "test"):
        split_dir = src_root / split
        if not split_dir.exists():
            continue

        xmls = {p.stem: p for p in split_dir.glob("*.xml")}
        for img in split_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            xml = xmls.get(img.stem)
            if xml is None:
                continue
            key = str(img.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((img, xml))
    return pairs


def link_or_copy(src: Path, dst: Path, mode: str):
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


def split_groups(groups, train_ratio: float, valid_ratio: float, seed: int):
    keys = list(groups)
    random.Random(seed).shuffle(keys)

    total = sum(len(groups[k]) for k in keys)
    target_train = int(total * train_ratio)
    target_valid = int(total * valid_ratio)

    result = {"train": [], "valid": [], "test": []}
    for key in keys:
        bucket = groups[key]
        if len(result["train"]) < target_train:
            result["train"].extend(bucket)
        elif len(result["valid"]) < target_valid:
            result["valid"].extend(bucket)
        else:
            result["test"].extend(bucket)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create a clean 80/10/10 Pascal VOC split from a Roboflow dataset."
    )
    parser.add_argument("--src", default="Accident.v2i.voc", help="Source VOC dataset root.")
    parser.add_argument(
        "--dst",
        default="Accident.v2i.voc_80_10_10",
        help="Destination dataset root.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink saves disk space; copy creates independent files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination split before recreating it.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        raise SystemExit(f"Source dataset not found: {src_root}")
    if dst_root.exists() and args.overwrite:
        shutil.rmtree(dst_root)
    if dst_root.exists() and any(dst_root.iterdir()):
        raise SystemExit(
            f"Destination already exists and is not empty: {dst_root}\n"
            "Use --overwrite if you want to recreate it."
        )

    pairs = collect_pairs(src_root)
    if not pairs:
        raise SystemExit(f"No image/XML pairs found under: {src_root}")

    groups = defaultdict(list)
    for pair in pairs:
        groups[source_key(pair[0])].append(pair)

    splits = split_groups(groups, args.train_ratio, args.valid_ratio, args.seed)
    for split_name, split_pairs in splits.items():
        for img, xml in split_pairs:
            link_or_copy(img, dst_root / split_name / img.name, args.mode)
            link_or_copy(xml, dst_root / split_name / xml.name, args.mode)

    total = sum(len(v) for v in splits.values())
    print(f"Created split at: {dst_root}")
    print(f"Total pairs : {total}")
    print(f"Source groups: {len(groups)}")
    for split_name in ("train", "valid", "test"):
        count = len(splits[split_name])
        pct = count / total * 100 if total else 0
        print(f"{split_name:5}: {count:6} pairs ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
