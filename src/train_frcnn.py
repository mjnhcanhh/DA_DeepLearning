import os
import torch
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# --- 1. Class YoloDataset (Giữ nguyên để đảm bảo tính nhất quán dữ liệu) ---
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = [f for f in sorted(os.listdir(img_dir)) if f.endswith(('.jpg', '.png'))]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        label_path = os.path.join(self.label_dir, self.imgs[idx].rsplit('.', 1)[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x_c, y_c, wb, hb = map(float, line.split())
                    xmin, xmax = (x_c - wb/2) * w, (x_c + wb/2) * w
                    ymin, ymax = (y_c - hb/2) * h, (y_c + hb/2) * h
                    xmin, xmax = max(0, xmin), min(w, xmax)
                    ymin, ymax = max(0, ymin), min(h, ymax)
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(cls) + 1)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, target

    def __len__(self):
        return len(self.imgs)

# --- 2. Hàm khởi tạo Model ---
def get_model(num_classes):
    # Sử dụng weights=None vì chúng ta sẽ load trọng số đã train từ file .pth
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 3. Hàm Train chính (CHẠY TIẾP TỪ 31 ĐẾN 50) ---
def train_faster_rcnn():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🚀 Đang chạy trên: {device}")

    # Đường dẫn (Kiểm tra kỹ các folder này trước khi chạy)
    train_img_dir = "Data/train/images"   
    train_label_dir = "Data/train/labels"
    save_path = "models/faster_rcnn_accident.pth"
    num_classes = 4 

    # Chuẩn bị Data
    dataset = YoloDataset(train_img_dir, train_label_dir)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), drop_last=True, pin_memory=True)

    model = get_model(num_classes).to(device)
    
    # --- NẠP LẠI TRỌNG SỐ EPOCH 30 ---
    if os.path.exists(save_path):
        print(f"♻️ Tìm thấy file trọng số tại Epoch 30. Đang nạp để cày tiếp...")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("⚠️ Không tìm thấy file trọng số cũ! Mô hình sẽ học lại từ đầu.")

    # Optimizer: Giữ Learning Rate thấp (0.001) để tinh chỉnh chính xác hơn ở giai đoạn cuối
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9)

    # Cấu hình Epoch
    start_epoch = 30  # Đã xong 30
    end_epoch = 50    # Chạy đến 50
    
    print(f"\n🏁 GIAI ĐOẠN NƯỚC RÚT: Từ Epoch {start_epoch + 1} đến {end_epoch}")
    
    for epoch in range(start_epoch, end_epoch): 
        model.train()
        epoch_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{end_epoch}", unit="batch")
        
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            pbar.set_postfix({'Loss': f"{losses.item():.4f}"})
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"✅ Xong Epoch {epoch+1}. Loss trung bình: {avg_loss:.4f}")

        # Lưu đè trọng số sau mỗi Epoch để tránh mất điện giữa chừng
        torch.save(model.state_dict(), save_path)
        print(f"💾 Đã lưu trọng số mới nhất vào: {save_path}\n")

    print(f"🎉 Chúc mừng Hùng! Đã hoàn thành cột mốc 50 Epochs.")

if __name__ == "__main__":
    # Đảm bảo thư mục models tồn tại
    if not os.path.exists("models"):
        os.makedirs("models")
    
    train_faster_rcnn()