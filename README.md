# 🚨 DA_DeepLearning — Hệ Thống Phát Hiện Tai Nạn Giao Thông

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![YOLOv12](https://img.shields.io/badge/Model-YOLOv12-green.svg)]()
[![CRNN](https://img.shields.io/badge/Model-CRNN-orange.svg)]()
[![SSD](https://img.shields.io/badge/Model-SSD-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> Đồ án tốt nghiệp — Hệ thống phát hiện tai nạn giao thông tự động sử dụng kết hợp ba mô hình học sâu: **YOLOv12**, **CRNN** và **SSD**, cho ra một kết quả dự đoán cuối cùng thông qua cơ chế kết hợp đầu ra (ensemble).

---

## 📋 Mục Lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Huấn luyện mô hình](#-huấn-luyện-mô-hình)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [Kết quả](#-kết-quả)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới Thiệu

Tai nạn giao thông là vấn đề nghiêm trọng, gây ra thiệt hại lớn về người và tài sản. Hệ thống này ứng dụng học sâu để **tự động phát hiện tai nạn qua camera/video**, từ đó cảnh báo kịp thời.

### Điểm nổi bật:
- **Kết hợp 3 mô hình** (YOLOv12 + CRNN + SSD) cho độ chính xác cao hơn từng mô hình riêng lẻ
- Phát hiện tai nạn theo **thời gian thực**
- Giao diện web trực quan qua **Flask** (`app.py`)
- Hỗ trợ đầu vào: **ảnh, video, webcam**

---

## 🏗 Kiến Trúc Hệ Thống

```
Input (Ảnh / Video)
        │
        ▼
┌───────────────────────────────────────────┐
│              Tiền xử lý                   │
│     (Resize, Normalize, Frame Extract)    │
└───────────┬───────────┬───────────────────┘
            │           │           │
            ▼           ▼           ▼
       ┌─────────┐ ┌─────────┐ ┌─────────┐
       │ YOLOv12 │ │  CRNN   │ │   SSD   │
       │(Phát hiện│ │(Trình tự│ │(Phát hiện│
       │ vật thể)│ │ thời gian│ │ vật thể)│
       └────┬────┘ └────┬────┘ └────┬────┘
            │           │           │
            └─────────┬─────────────┘
                       ▼
            ┌─────────────────────┐
            │  Ensemble / Fusion  │
            │  (Kết hợp đầu ra)  │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │   Kết Quả Cuối Cùng │
            │  Có / Không tai nạn │
            └─────────────────────┘
```

### Vai trò của từng mô hình:

| Mô hình | Vai trò | Đặc điểm |
|---------|---------|-----------|
| **YOLOv12** | Phát hiện vật thể theo thời gian thực | Tốc độ cao, độ chính xác tốt với vật thể di chuyển |
| **CRNN** | Phân tích chuỗi thời gian từ video | Khai thác thông tin thời gian giữa các frame liên tiếp |
| **SSD** | Phát hiện vật thể đa tỷ lệ | Mạnh với vật thể nhiều kích thước khác nhau |

---

## 📁 Cấu Trúc Thư Mục

```
DA_DeepLearning/
├── Models/                  # Trọng số đã huấn luyện (.pt, .pth, .h5)
│   ├── yolov12_accident.pt
│   ├── crnn_accident.pth
│   └── ssd_accident.pth
│
├── runs/                    # Kết quả huấn luyện (log, metrics, charts)
│   └── train/
│       ├── yolov12/
│       ├── crnn/
│       └── ssd/
│
├── src/                     # Mã nguồn chính
│   ├── models/              # Định nghĩa kiến trúc mô hình
│   │   ├── yolov12.py
│   │   ├── crnn.py
│   │   └── ssd.py
│   ├── train/               # Script huấn luyện từng mô hình
│   ├── predict/             # Script dự đoán / inference
│   ├── ensemble.py          # Kết hợp đầu ra 3 mô hình
│   └── utils.py             # Hàm tiện ích chung
│
├── templates/               # Giao diện web (HTML)
│   └── index.html
│
├── venv/                    # Môi trường ảo Python
├── app.py                   # Ứng dụng Flask chính (28 KB)
├── requirements.txt         # Các thư viện cần thiết
└── .gitignore
```

---

## ⚙️ Yêu Cầu Hệ Thống

- **Python** >= 3.8
- **GPU** khuyến nghị: NVIDIA với CUDA >= 11.3 (có thể chạy CPU nhưng chậm hơn)
- **RAM** >= 8 GB
- **Dung lượng** >= 5 GB (dataset + models)

---

## 🚀 Cài Đặt

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/DA_DeepLearning.git
cd DA_DeepLearning
```

### 2. Tạo môi trường ảo và cài thư viện

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Tải trọng số mô hình (nếu chưa có)

Đặt các file trọng số vào thư mục `Models/`:
```
Models/
├── yolov12_accident.pt
├── crnn_accident.pth
└── ssd_accident.pth
```

---

## 🏋️ Huấn Luyện Mô Hình

> ☁️ **Toàn bộ quá trình huấn luyện được thực hiện trên [Google Colab](https://colab.research.google.com/) với GPU T4/A100 miễn phí.**

### Quy trình huấn luyện trên Colab:

1. Upload dataset lên **Google Drive**
2. Mở notebook tương ứng trên Colab
3. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Cài đặt thư viện:

```python
!pip install ultralytics torch torchvision opencv-python
```

5. Chạy huấn luyện từng mô hình:

```python
# YOLOv12
!yolo task=detect mode=train model=yolov12n.pt data=/content/drive/MyDrive/accident.yaml epochs=100 batch=16

# CRNN
!python src/train/train_crnn.py --data /content/drive/MyDrive/data/ --epochs 50

# SSD
!python src/train/train_ssd.py --data /content/drive/MyDrive/data/ --epochs 100
```

6. Sau khi train xong, **tải file trọng số `.pt` / `.pth` về** và đặt vào thư mục `Models/` của project.

> 📁 Notebook Colab: [`notebooks/train_yolov12.ipynb`](notebooks/), [`notebooks/train_crnn.ipynb`](notebooks/), [`notebooks/train_ssd.ipynb`](notebooks/)

Kết quả huấn luyện (loss, accuracy, confusion matrix) được lưu tự động vào thư mục `runs/`.

---

## ▶️ Chạy Ứng Dụng

### Khởi động server Flask

```bash
python app.py
```

Mở trình duyệt và truy cập: **http://localhost:5000**

### Tính năng giao diện:
- 📷 Upload ảnh/video để phân tích
- 🎥 Kết nối webcam phát hiện thời gian thực
- 📊 Hiển thị xác suất dự đoán từng mô hình và kết quả tổng hợp

---

## 📊 Kết Quả

| Mô hình | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| YOLOv12 | ~xx% | ~xx% | ~xx% | ~xx% |
| CRNN | ~xx% | ~xx% | ~xx% | ~xx% |
| SSD | ~xx% | ~xx% | ~xx% | ~xx% |
| **Ensemble** | **~xx%** | **~xx%** | **~xx%** | **~xx%** |

> 📝 *Cập nhật kết quả thực tế sau khi huấn luyện xong.*

---

## 🛠 Công Nghệ Sử Dụng

- [PyTorch](https://pytorch.org/) — Framework học sâu chính
- [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics) — Object detection
- [Flask](https://flask.palletsprojects.com/) — Web framework
- [OpenCV](https://opencv.org/) — Xử lý ảnh và video
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) — Xử lý dữ liệu

---

## 👨‍💻 Tác Giả

| Thông tin | Chi tiết |
|-----------|----------|
| **Họ tên** | *(Tên của bạn)* |
| **MSSV** | *(Mã số sinh viên)* |
| **Trường** | *(Tên trường)* |
| **GVHD** | *(Tên giáo viên hướng dẫn)* |
| **Năm** | 2026 |

---

## 📄 Giấy Phép

Dự án này được phân phối theo giấy phép [MIT](LICENSE).

---

<p align="center">⭐ Nếu thấy hữu ích, hãy cho dự án một star!</p>
