<<<<<<< HEAD
# 🚨 Mô hình AI nhận diện tai nạn và hỗ trợ phản ứng khẩn cấp thời gian thực

## 📁 Cấu trúc dự án

```
├── data/               # Dataset (images, videos mẫu)
├── models/             # File trọng số .pt hoặc .h5 đã huấn luyện
├── src/
=======
Mô hình AI nhận diện tai nạn và hỗ trợ phản ứng khẩn cấp thời gian thực
Giới thiệu (Overview)
Dự án được phát triển bởi nhóm sinh viên HUIT. Hệ thống sử dụng các mô hình Deep Learning tiên tiến để giám sát khu vực đường đi bộ/vỉa hè, tự động phát hiện các hành vi vi phạm giao thông và đặc biệt là nhận diện tai nạn.
- Khi có sự cố, hệ thống sẽ ngay lập tức:
 - Chụp ảnh hiện trường làm bằng chứng.
 - Định vị vị trí xảy ra tai nạn.
 - Gửi thông báo khẩn cấp (Email/Telegram) kèm hình ảnh tới cơ quan chức năng (Công an, Bệnh viện).
Công nghệ sử dụng
- Ngôn ngữ: Python 3.9+
- Deep Learning Framework: PyTorch, Ultralytics (YOLOv11/v8).
- Thị giác máy tính: OpenCV, MediaPipe.
- Giao diện: Streamlit / Flask.
-Database & Log: SQLite / CSV để lưu trữ lịch sử tai nạn.
Dữ liệu 
Cấu trúc dự án
DA_DeepLearning/
├── data/               # Chứa dataset (images, videos mẫu)
├── models/             # Chứa file trọng số .pt hoặc .h5 đã huấn luyện
├── src/                # Mã nguồn chính
>>>>>>> d0f5d888907ac2602b8b003eafbf0916ba1717f2
│   ├── detection.py    # Module nhận diện vật thể & tai nạn
│   ├── tracking.py     # Module theo dõi quỹ đạo đối tượng
│   ├── alert_system.py # Module gửi thông báo (Email/API)
│   └── utils.py        # Các hàm bổ trợ xử lý hình ảnh
<<<<<<< HEAD
├── app.py              # File chạy giao diện Dashboard (Streamlit)
├── requirements.txt    # Danh sách thư viện cần cài đặt
└── README.md
```

## 🤖 3 Thuật toán được so sánh

| Hạng | Thuật toán | mAP@0.5 | FPS | Model Size |
|------|-----------|---------|-----|------------|
| 🏆 #1 | **YOLOv8** | **94.7%** | 47.2 | 22.5 MB |
| 🥈 #2 | SSD MobileNet | 87.3% | **62.8** | **14.2 MB** |
| 🥉 #3 | Faster R-CNN | 91.5% | 18.4 | 167 MB |

## ⚡ Cài đặt & Chạy

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy dashboard
streamlit run app.py
```

## 📊 Biểu đồ so sánh

Dashboard tự động hiển thị **3 biểu đồ so sánh**:

1. **🕸️ Radar Chart** — Tổng quan đa chiều (mAP, Precision, Recall, F1)
2. **📊 Grouped Bar Chart** — So sánh từng chỉ số accuracy
3. **🎯 Scatter Plot** — Trade-off: Tốc độ (FPS) vs Độ chính xác (mAP)

> YOLOv8 xếp **#1** và hiển thị **đầu tiên** trong mọi biểu đồ
=======
├── app.py              # File chạy giao diện Dashboard
├── requirements.txt    # Danh sách các thư viện cần cài đặt
└── README.md
Hướng dẫn cài đặt
Bước 1: Clone repository
git clone https://github.com/your-username/sers-accident-detection.git
cd sers-accident-detection
Bước 2: Cài đặt môi trường
pip install -r requirements.txt
Bước 3: Chạy hệ thống
python app.pyS
Hướng dẫn sử dụng 
Kết quả mong đợi 
>>>>>>> d0f5d888907ac2602b8b003eafbf0916ba1717f2
