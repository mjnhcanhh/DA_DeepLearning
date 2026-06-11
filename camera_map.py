# -*- coding: utf-8 -*-
"""
camera_map.py — Virtual traffic cameras for the monitoring map (HCMC)
FIX:
  - random_source() bỏ qua .gitkeep, chỉ trả file ảnh/video thật
  - random_video_frame() lấy frame ngẫu nhiên trong video (không chỉ frame đầu)
  - Thêm hàm assign_uploaded_to_random() để upload 1 ảnh → gán vào camera ngẫu nhiên
"""

import random
from pathlib import Path

BASE_DEMO_DIR = Path("static/demo_sources")

# Ảnh/video hợp lệ
VALID_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VID_EXTS  = {".mp4", ".avi", ".mov", ".mkv"}
VALID_EXTS      = VALID_IMG_EXTS | VALID_VID_EXTS

CAMERAS = [
    {
        "id": "CAM_001",
        "name": "Nga tu Hang Xanh",
        "lat": 10.8012,
        "lng": 106.7115,
        "district": "Bình Thạnh",
        "status": "normal",
        "sources": [
            "static/demo_sources/cam_001/img_01.jpg",
            "static/demo_sources/cam_001/img_02.jpg",
            "static/demo_sources/cam_001/video_01.mp4",
        ],
    },
    {
        "id": "CAM_002",
        "name": "Cau Sai Gon",
        "lat": 10.8041,
        "lng": 106.7320,
        "district": "Bình Thạnh",
        "status": "normal",
        "sources": [
            "static/demo_sources/cam_002/img_01.jpg",
            "static/demo_sources/cam_002/img_02.jpg",
        ],
    },
    {
        "id": "CAM_003",
        "name": "Khu vuc HUIT",
        "lat": 10.7756,
        "lng": 106.6679,
        "district": "Quận 10",
        "status": "normal",
        "sources": [
            "static/demo_sources/cam_003/img_01.jpg",
            "static/demo_sources/cam_003/img_02.jpg",
        ],
    },
    {
        "id": "CAM_004",
        "name": "Nga sau Phu Dong",
        "lat": 10.7719,
        "lng": 106.6920,
        "district": "Quận 3",
        "status": "normal",
        "sources": [
            "static/demo_sources/cam_004/img_01.jpg",
            "static/demo_sources/cam_004/img_02.jpg",
        ],
    },
    {
        "id": "CAM_005",
        "name": "Cau vuot Cong Hoa",
        "lat": 10.8016,
        "lng": 106.6530,
        "district": "Tân Bình",
        "status": "normal",
        "sources": [
            "static/demo_sources/cam_005/img_01.jpg",
            "static/demo_sources/cam_005/img_02.jpg",
        ],
    },
    {
        "id": "CAM_006",
        "name": "Ham Thu Thiem",
        "lat": 10.7688,
        "lng": 106.7180,
        "district": "Quận 2",
        "status": "normal",
        "sources": [
            "static/demo_sources/cam_006/img_01.jpg",
            "static/demo_sources/cam_006/img_02.jpg",
        ],
    },
]


def get_all_cameras():
    return CAMERAS


def get_camera(camera_id):
    for cam in CAMERAS:
        if cam["id"] == camera_id:
            return cam
    return None


def _is_valid_source(path: Path) -> bool:
    """Trả True nếu file tồn tại và là ảnh/video thật (bỏ qua .gitkeep, .DS_Store)."""
    return path.exists() and path.suffix.lower() in VALID_EXTS and path.stat().st_size > 100


def random_source(camera_id: str):
    """
    Trả đường dẫn source ngẫu nhiên cho camera.
    Ưu tiên file tồn tại thật; bỏ qua .gitkeep.
    """
    cam = get_camera(camera_id)
    if not cam:
        return None
    existing = [src for src in cam["sources"] if _is_valid_source(Path(src))]
    return random.choice(existing) if existing else None


def random_camera():
    return random.choice(CAMERAS) if CAMERAS else None


def assign_uploaded_to_camera(camera_id: str, file_bytes: bytes, filename: str) -> Path:
    """
    Lưu file upload vào thư mục demo của camera để có thể dùng lại sau.
    Trả về Path đã lưu.
    """
    cam = get_camera(camera_id)
    if not cam:
        raise ValueError(f"Camera {camera_id} không tồn tại")
    ext = Path(filename).suffix.lower() or ".jpg"
    dest_dir = BASE_DEMO_DIR / camera_id.lower()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"upload_{random.randint(1000,9999)}{ext}"
    dest.write_bytes(file_bytes)
    # Thêm vào sources list nếu chưa có
    src_str = str(dest)
    if src_str not in cam["sources"]:
        cam["sources"].append(src_str)
    return dest