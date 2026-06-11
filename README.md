# AI Accident Detection Dashboard v4

## Overview

AI Accident Detection Dashboard v4 is a Flask-based web application for detecting traffic accidents from images, videos, webcam frames, and simulated traffic cameras. The system combines deep learning inference, model comparison, ensemble voting, a camera monitoring map, incident creation, and emergency dispatch simulation.

The project is designed for a deep learning course/demo. Heavy trained model weights and datasets are not required in the Git repository; they can be added locally when running full inference.

## Main Features

- Detect traffic accidents from uploaded images.
- Detect accident evidence from uploaded videos.
- Support webcam/camera frame inference.
- Compare multiple AI models on the same input image.
- Combine model outputs with ensemble voting.
- Display detection results with bounding boxes, confidence scores, processing time, and status labels.
- Mark wrong predictions and calculate a quick accuracy score on demo images.
- Simulate a traffic camera network in Ho Chi Minh City.
- Scan all virtual cameras and update camera status on the map.
- Create accident incidents automatically when the ensemble confirms an accident.
- Dispatch nearby traffic police and ambulance units.
- Draw dispatch routes on the map using OSRM when available, with fallback to straight-line routing.

## Project Structure

```text
DOAN/
|
|-- app.py                  # Main Flask application and API routes
|-- models.py               # AI model configuration and model loading logic
|-- detection.py            # Detection post-processing, bounding boxes, image encoding
|-- ensemble.py             # Ensemble decision logic
|-- camera_map.py           # Virtual camera list, camera sources, upload assignment
|-- emergency.py            # Incident registry and dispatch planning
|-- requirements.txt        # Python dependencies
|-- Train_Yolov12.ipynb     # YOLO training notebook
|
|-- templates/
|   |-- index.html          # Main web interface
|
|-- static/
|   |-- css/
|   |   |-- style.css
|   |   |-- patch.css
|   |
|   |-- js/
|       |-- main.js         # Upload, detection, webcam, UI tabs
|       |-- benchmark.js    # Model benchmark UI
|       |-- charts.js       # Charts and statistics
|       |-- map.js          # Camera map, scanning, dispatch routes
|       |-- emergency.js    # Incident list and dispatch actions
|
|-- src/                    # Supporting detection/tracking/alert modules
|-- code_TrainCNN/          # Faster R-CNN training code
|-- LocDuLieu/              # Dataset cleaning/splitting/inference helper scripts
```

## Requirements

- Python 3.8 or newer
- Windows, Linux, or macOS
- Recommended: NVIDIA GPU with CUDA for faster inference
- Minimum RAM: 8 GB recommended

The app can still start without model weights, but real AI prediction requires the trained weight files to be placed in the expected paths.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd DOAN
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model Weights

Model weights are intentionally excluded from Git because they are large. To run full inference, place the trained weights in one of the paths configured in `models.py`.

Common expected paths:

```text
Models/yolov12.pt
Models/fast_cnn_model.pth
runs/detect/train8/weights/last.pt
```

If model files are missing, the dashboard can still open, but model status will show that the corresponding model is not loaded.

## Run the Application

Start the Flask server:

```bash
python app.py
```

Open the browser:

```text
http://localhost:5000
```

The app runs on:

```text
Host: 0.0.0.0
Port: 5000
```

## How to Use

1. Open the dashboard at `http://localhost:5000`.
2. Upload an image or video for accident detection.
3. Select the model or use the benchmark panel to compare models.
4. Review bounding boxes, confidence, inference time, and prediction status.
5. Open the camera map tab to scan virtual traffic cameras.
6. If an accident is detected, check the incident panel and dispatch emergency units.

## Optimized Camera Workflow

The camera module simulates a traffic monitoring center:

- `camera_map.py` defines virtual cameras with IDs, names, districts, coordinates, and demo media sources.
- `static/js/map.js` renders camera markers on a Leaflet map.
- `/scan_cameras` scans all cameras and updates each camera status.
- `/camera_health` returns the latest health data, including latency, FPS, scan time, status, and cascade details.
- `/scan_random_camera_image` uploads an image/video, assigns it to a random camera, and runs detection immediately.

The camera scan uses a cascade strategy:

1. Fast models run first, such as SSD and YOLOv12.
2. Faster R-CNN only runs when the fast models detect a suspicious accident pattern or confidence passes a threshold.
3. This reduces processing time for normal camera scenes while still verifying dangerous cases.

When an accident is confirmed:

- The camera marker changes to accident status.
- An incident is created.
- The incident receives an ensemble score, vote count, priority score, and evidence image/video.
- The user can dispatch traffic police and ambulance units.
- The map draws dispatch routes. If OSRM is available, it draws real road routes; otherwise it falls back to straight-line routes.

## Training Notes

YOLO training is kept in:

```text
Train_Yolov12.ipynb
```

Faster R-CNN/CNN training code is kept in:

```text
code_TrainCNN/
```

Dataset cleaning and splitting utilities are kept in:

```text
LocDuLieu/
```

Datasets, generated training runs, and trained weights should not be pushed to Git unless required by the instructor.

---

# Tài liệu tiếng Việt

## Giới thiệu

Đây là dashboard phát hiện tai nạn giao thông bằng AI, được xây dựng bằng Flask. Hệ thống cho phép người dùng tải ảnh/video, chạy nhận diện bằng nhiều mô hình, so sánh kết quả, theo dõi mạng camera ảo trên bản đồ và tạo quy trình cảnh báo - điều phối khi phát hiện tai nạn.

Mục tiêu của đồ án:

- Phát hiện tai nạn giao thông từ ảnh, video hoặc camera/webcam.
- Hiển thị kết quả trực quan bằng bounding box, mức cảnh báo và độ tin cậy.
- So sánh kết quả giữa các mô hình AI.
- Mô phỏng hệ thống camera giao thông tại TP.HCM.
- Tạo sự cố tai nạn và điều phối đơn vị phản ứng nhanh.

## Cấu trúc dự án

```text
DOAN/
|
|-- app.py                  # Flask app, route API, xử lý detect ảnh/video/camera
|-- models.py               # Cấu hình và load các mô hình AI
|-- detection.py            # Xử lý kết quả detect, vẽ bounding box, encode ảnh
|-- ensemble.py             # Tổng hợp kết quả nhiều model để ra quyết định cuối
|-- camera_map.py           # Danh sách camera ảo, nguồn demo, gán file upload vào camera
|-- emergency.py            # Tạo incident, tính đơn vị gần nhất, điều phối cứu hộ
|-- requirements.txt        # Thư viện cần cài đặt
|-- Train_Yolov12.ipynb     # Notebook huấn luyện YOLO
|
|-- templates/
|   |-- index.html          # Giao diện chính
|
|-- static/
|   |-- css/
|   |   |-- style.css       # Style chính
|   |   |-- patch.css       # Style bổ sung
|   |
|   |-- js/
|       |-- main.js         # Xử lý upload, detect, webcam, tab UI
|       |-- benchmark.js    # So sánh model trên cùng một ảnh
|       |-- charts.js       # Biểu đồ thống kê/benchmark
|       |-- map.js          # Bản đồ camera, scan camera, route điều phối
|       |-- emergency.js    # Danh sách sự cố, cập nhật trạng thái, dispatch
|
|-- src/                    # Module hỗ trợ detect/tracking/alert
|-- code_TrainCNN/          # Code huấn luyện Faster R-CNN
|-- LocDuLieu/              # Script lọc dữ liệu, chia dữ liệu, test inference
```

## Cài đặt

### 1. Clone source code

```bash
git clone <your-repository-url>
cd DOAN
```

### 2. Tạo môi trường ảo

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Cài thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Cấu hình model

Các file model không nên đưa lên Git vì dung lượng lớn. Khi cần chạy nhận diện thật, đặt model vào đúng đường dẫn được cấu hình trong `models.py`.

Các đường dẫn thường dùng:

```text
Models/yolov12.pt
Models/fast_cnn_model.pth
runs/detect/train8/weights/last.pt
```

Nếu chưa có model, giao diện vẫn có thể mở nhưng phần trạng thái model sẽ báo chưa load được model tương ứng.

## Chạy chương trình

Khởi động Flask server:

```bash
python app.py
```

Sau đó mở trình duyệt:

```text
http://localhost:5000
```

Ứng dụng chạy với:

```text
Host: 0.0.0.0
Port: 5000
```

## Các chức năng chính

### 1. Nhận diện tai nạn từ ảnh

Người dùng có thể tải một hoặc nhiều ảnh lên dashboard. Hệ thống sẽ đọc ảnh, đưa qua mô hình AI đang được chọn và trả về:

- Ảnh đã vẽ bounding box.
- Nhãn kết quả: bình thường, cảnh báo hoặc tai nạn.
- Độ tin cậy của mô hình.
- Số lượng đối tượng phát hiện.
- Thời gian xử lý.

Kết quả được chia thành các nhóm để dễ theo dõi:

- Bình thường: không phát hiện dấu hiệu tai nạn.
- Dự đoán nguy cơ: có dấu hiệu cảnh báo.
- Tai nạn phát hiện: mô hình xác định có tai nạn.

### 2. Nhận diện từ video

Hệ thống hỗ trợ tải video giao thông lên để quét khung hình. Thay vì chỉ xử lý một ảnh tĩnh, video được đọc theo frame và AI sẽ phân tích các frame đại diện. Kết quả giúp đánh giá tình huống tai nạn trong clip, đồng thời hiển thị ảnh minh chứng và thông tin thời gian xử lý.

### 3. Webcam / camera trực tiếp

Dashboard có chức năng xử lý từ webcam/camera. Người dùng có thể bật camera, lấy frame và chạy nhận diện. Chức năng này dùng để mô phỏng bài toán giám sát thời gian gần thực trong hệ thống giao thông.

### 4. Chọn và so sánh mô hình AI

Hệ thống được thiết kế để làm việc với nhiều mô hình:

- SSD
- Faster R-CNN
- YOLOv12

Người dùng có thể chọn mô hình đang sử dụng trên giao diện. Ngoài ra, màn hình benchmark cho phép chạy cùng một ảnh qua nhiều mô hình để so sánh:

- Kết quả phát hiện của từng mô hình.
- Độ tin cậy.
- Thời gian xử lý.
- FPS.
- Số bounding box.
- Trạng thái mô hình đã load hay chưa.

### 5. Ensemble kết quả

Phần ensemble tổng hợp kết quả từ nhiều mô hình để đưa ra quyết định cuối cùng. Thay vì chỉ phụ thuộc vào một mô hình, hệ thống tính:

- Số mô hình bỏ phiếu có tai nạn.
- Điểm tin cậy tổng hợp.
- Lý do đưa ra kết luận.
- Mức ưu tiên xử lý sự cố.

Chức năng này giúp kết quả ổn định hơn khi từng mô hình có thể dự đoán sai trong một số trường hợp.

### 6. Đánh dấu kết quả sai và tính độ chính xác

Sau khi xử lý ảnh, người dùng có thể đánh dấu những trường hợp mô hình dự đoán sai. Từ đó dashboard tính lại tỉ lệ đúng/sai trên tập ảnh đã kiểm tra:

- Ảnh không bị đánh dấu: xem như mô hình đúng.
- Ảnh bị đánh dấu: xem như mô hình sai.
- Hiển thị số ảnh đúng, sai và tổng số ảnh.
- Tính phần trăm chính xác.

Chức năng này dùng để kiểm tra nhanh chất lượng mô hình trên tập ảnh demo.

## Chức năng camera đã tối ưu

Phần camera là phần mô phỏng trung tâm giám sát giao thông. Hệ thống không chỉ detect ảnh riêng lẻ mà còn gắn kết kết quả AI với từng vị trí camera trên bản đồ.

### 1. Mạng camera ảo tại TP.HCM

File `camera_map.py` khai báo sẵn nhiều camera giao thông ảo với tọa độ gần các khu vực quen thuộc:

- Ngã tư Hàng Xanh.
- Cầu Sài Gòn.
- Khu vực HUIT.
- Ngã sáu Phù Đổng.
- Cầu vượt Cộng Hòa.
- Hầm Thủ Thiêm.

Mỗi camera có:

- Mã camera.
- Tên vị trí.
- Quận/khu vực.
- Kinh độ, vĩ độ.
- Danh sách ảnh/video demo làm nguồn quét.

### 2. Hiển thị camera trên bản đồ

File `static/js/map.js` dùng Leaflet để vẽ bản đồ và marker camera. Mỗi camera được hiển thị bằng marker riêng. Khi bấm vào marker, người dùng xem được:

- Mã camera.
- Tên vị trí.
- Trạng thái hiện tại.
- Điểm ensemble.
- Số vote của các mô hình.
- Mức ưu tiên nếu có tai nạn.
- Mã sự cố nếu incident đã được tạo.

Nếu thư viện bản đồ không tải được, hệ thống có chế độ offline map: vẫn hiển thị các điểm camera trên một bản đồ giả lập, giúp giao diện không bị trống.

### 3. Quét toàn bộ camera

Nút quét camera gọi route `/scan_cameras`. Hệ thống sẽ lần lượt:

1. Lấy nguồn ảnh/video ngẫu nhiên của từng camera.
2. Đọc ảnh hoặc lấy frame trong video.
3. Chạy AI trên frame đó.
4. Tổng hợp kết quả bằng ensemble.
5. Cập nhật màu marker trên bản đồ.
6. Tạo incident nếu phát hiện tai nạn.
7. Cập nhật bảng health của camera.

Trạng thái camera gồm:

- Normal: camera không phát hiện tai nạn.
- Accident: camera phát hiện tai nạn.
- No source: camera chưa có ảnh/video demo.
- Source error: file demo bị lỗi hoặc không đọc được.

### 4. Cascade tối ưu tốc độ

Phần camera đã được tối ưu bằng cơ chế cascade. Khi quét camera, hệ thống không mặc định chạy tất cả mô hình nặng ngay từ đầu.

Luôn chạy nhóm mô hình nhanh trước:

- SSD
- YOLOv12

Sau đó Faster R-CNN chỉ chạy khi:

- Mô hình nhanh có dấu hiệu phát hiện tai nạn.
- Hoặc độ tin cậy vượt ngưỡng nghi ngờ.

Lợi ích:

- Giảm thời gian xử lý khi camera bình thường.
- Vẫn có bước xác minh khi có tình huống nguy hiểm.
- Phù hợp hơn với bài toán giám sát nhiều camera.

Thông tin cascade được hiển thị trên giao diện:

- Các mô hình đã chạy.
- Faster R-CNN có được kích hoạt hay không.
- Latency.
- FPS.
- Số vote của nhóm mô hình nhanh.
- Điểm tin cậy cao nhất.

### 5. Camera health panel

Hệ thống có bảng theo dõi sức khỏe camera thông qua route `/camera_health`. Mỗi camera được lưu lại thông tin lần quét gần nhất:

- Camera online hay offline.
- Trạng thái mới nhất.
- Thời gian quét cuối.
- FPS.
- Latency.
- Cascade đã chạy như thế nào.
- Nguồn ảnh/video đã dùng.

Bảng health giúp người quản trị biết camera nào vừa được quét, camera nào đang có tai nạn và tốc độ xử lý của hệ thống.

### 6. Upload ảnh/video vào camera ngẫu nhiên

Người dùng có thể upload một ảnh hoặc video, hệ thống sẽ gán file đó vào một camera ngẫu nhiên thông qua route `/scan_random_camera_image`.

Quy trình:

1. Người dùng chọn file.
2. Hệ thống chọn một camera bất kỳ.
3. File được lưu vào `static/demo_sources/cam_xxx/`.
4. AI chạy ngay trên file vừa upload.
5. Kết quả được gán trực tiếp lên camera đó trên bản đồ.
6. Nếu là video, giao diện có thể hiển thị replay bằng chứng.

Chức năng này giúp demo linh hoạt: chỉ cần upload một ảnh/video tai nạn, hệ thống sẽ mô phỏng như camera tại hiện trường vừa ghi nhận sự cố.

### 7. Tạo incident khi phát hiện tai nạn

Khi ensemble kết luận có tai nạn, `emergency.py` tạo một incident mới. Incident gồm:

- Mã sự cố.
- Camera phát hiện.
- Vị trí tai nạn.
- Điểm ensemble.
- Số mô hình đồng thuận.
- Điểm ưu tiên.
- Ảnh bằng chứng.
- Video bằng chứng nếu có.
- Trạng thái xử lý.

Incident được hiển thị trong danh sách sự cố và trên bản đồ.

### 8. Điều phối CSGT và cứu thương

Hệ thống mô phỏng quy trình phản ứng nhanh. Khi bấm nút điều phối, route `/incident/<id>/dispatch` sẽ:

- Chọn đơn vị CSGT gần vị trí tai nạn.
- Chọn đơn vị cứu thương gần vị trí tai nạn.
- Cập nhật trạng thái incident thành `DISPATCHED`.
- Ghi log thời gian điều phối.
- Hiển thị kế hoạch điều phối trên giao diện.

Người dùng có thể thấy:

- Đơn vị CSGT được chọn.
- Đơn vị cứu thương được chọn.
- Khoảng cách ước tính.
- ETA ước tính.
- Hotline của từng đơn vị.

### 9. Vẽ tuyến đường điều phối trên bản đồ

Sau khi dispatch, `map.js` vẽ tuyến đường từ đơn vị phản ứng nhanh đến vị trí tai nạn:

- Tuyến CSGT có màu riêng.
- Tuyến cứu thương có màu riêng.
- Nếu gọi được OSRM, hệ thống vẽ tuyến đường bộ thật.
- Nếu không gọi được OSRM, hệ thống fallback sang đường chim bay.

Mỗi tuyến có popup hiển thị:

- Đơn vị xuất phát.
- Camera/địa điểm tai nạn.
- Loại tuyến.
- Khoảng cách.
- Thời gian dự kiến.

### 10. Cảnh báo trực quan và âm thanh

Khi camera phát hiện tai nạn:

- Marker camera đổi sang trạng thái accident.
- Popup hiện thông tin sự cố.
- Nếu upload file và phát hiện tai nạn, giao diện phát âm thanh cảnh báo.
- Danh sách incident được cập nhật ngay.

Chức năng này giúp dashboard giống hệ thống giám sát thực tế hơn, không chỉ hiện kết quả AI mà còn có luồng xử lý sau phát hiện.

## Giao diện

Giao diện dashboard được thiết kế theo phong cách sáng, rõ ràng, tập trung vào thao tác:

- Tab upload/detect ảnh.
- Tab benchmark model.
- Tab camera map.
- Khu vực incident/emergency.
- Bảng health camera.
- Biểu đồ và thống kê.

Màu sắc được dùng để phân biệt nhanh:

- Xanh lá: bình thường.
- Vàng/cam: cảnh báo.
- Đỏ: tai nạn.
- Xanh dương: điều phối/CSGT.

## Ghi chú khi nộp code

Nếu chỉ nộp source code, không cần nộp các file model nặng như `.pt`, `.pth`, dataset train, thư mục `venv`, cache Python hoặc file demo media lớn. Khi chạy thực tế, cần bổ sung model vào đúng đường dẫn cấu hình trong `models.py`.

File `templates/map_view.html` không có trong project hiện tại. Giao diện chính vẫn nằm trong `templates/index.html`; nếu không dùng route `/map_view` thì không ảnh hưởng đến trang dashboard chính.
