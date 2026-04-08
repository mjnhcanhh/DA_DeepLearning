# Mô Hình AI Nhận Diện Tai Nạn & Hỗ Trợ Phản Ứng Khẩn Cấp

## Giới thiệu

Dự án được phát triển nhằm xây dựng hệ thống giám sát giao thông thông minh sử dụng Deep Learning.

Hệ thống có khả năng:

- Phát hiện tai nạn giao thông theo thời gian thực
- Chụp ảnh hiện trường
- Xác định vị trí xảy ra sự cố
- Gửi cảnh báo (Email/API) đến cơ quan chức năng

---

## Công nghệ sử dụng

- Ngôn ngữ: Python 3.10+
- Deep Learning: PyTorch, Ultralytics (YOLOv8)
- Computer Vision: OpenCV
- Tracking: DeepSORT
- Giao diện: Streamlit
- Xử lý dữ liệu: NumPy, Pandas, Matplotlib, Scikit-learn

---

## Cấu trúc dự án

DA_DeepLearning/
├── data/ # Dataset (images, videos mẫu)
├── models/ # File trọng số (.pt)
├── runs/ # Kết quả training YOLO
├── src/
│ ├── detection.py # Nhận diện tai nạn (YOLO)
│ ├── tracking.py # Theo dõi đối tượng
│ ├── alert_system.py # Gửi cảnh báo
│ └── utils.py # Hàm hỗ trợ
├── app.py # Dashboard (Streamlit)
├── best.pt # Model đã train
├── requirements.txt
└── README.md

---

## Hướng dẫn cài đặt

### Bước 1: Tạo môi trường (khuyến nghị)

conda create -n ai_env python=3.10  
conda activate ai_env

---

### Bước 2: Cài thư viện

pip install -r requirements.txt

---

## Hướng dẫn chạy

### 🔹 Chạy Dashboard (Khuyên dùng)

streamlit run app.py

👉 Mở trình duyệt tại:  
http://localhost:8501

---

### Chạy AI trực tiếp

python src/detection.py

---

## Mô hình sử dụng (dự tính)

| Thuật toán    | mAP@0.5 | FPS  | Size   |
| ------------- | ------- | ---- | ------ |
| YOLOv8        | 94.7%   | 47.2 | 22.5MB |
| SSD MobileNet | 87.3%   | 62.8 | 14.2MB |
| Faster R-CNN  | 91.5%   | 18.4 | 167MB  |

---

## Chức năng chính

- Nhận diện tai nạn từ video/camera
- Tracking đối tượng
- Gửi cảnh báo khi phát hiện sự cố
- Dashboard hiển thị kết quả

---

## Kết quả mong đợi

- Hiển thị bounding box đối tượng
- Cảnh báo khi phát hiện tai nạn
- Lưu ảnh sự kiện
- Thống kê trên dashboard

---

## Câu hỏi thường gặp

### Q: Chạy không thấy gì?

Kiểm tra:

- Model `best.pt` có tồn tại không
- Có video/camera input không

---

### Q: Lỗi thư viện?

Thử:

pip install --upgrade pip  
pip install -r requirements.txt

---

### Q: Chạy chậm?

Gợi ý:

- Dùng GPU (CUDA)
- Giảm độ phân giải video

---

## Nhóm phát triển

Nhóm 15: Hưng-Nhân-Cảnh-Nhật
