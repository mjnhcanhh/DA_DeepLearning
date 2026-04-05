import os
import torch
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

# Định nghĩa tên các lớp để hiển thị cho chuyên nghiệp
LABELS_MAP = {
    1: "Xe",
    2: "Tai nan",
    3: "Nguoi"
}

def test_inference(img_path, model, device):
    img = cv2.imread(img_path)
    if img is None: return None
    
    img_show = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    for i in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][i].item()
        # Ngưỡng 0.25 để lọc bớt nhiễu nhưng vẫn giữ được các vật thể mờ
        if score > 0.25: 
            box = predictions[0]['boxes'][i].cpu().numpy().astype(int)
            label_id = predictions[0]['labels'][i].item()
            
            # Lấy tên tiếng Việt từ map, nếu không có thì hiện ID
            label_name = LABELS_MAP.get(label_id, f"ID {label_id}")
            
            # Vẽ khung xanh
            cv2.rectangle(img_show, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Vẽ nền cho chữ để dễ đọc
            label_text = f"{label_name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_show, (box[0], box[1] - 20), (box[0] + w, box[1]), (0, 255, 0), -1)
            cv2.putText(img_show, label_text, (box[0], box[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img_show

if __name__ == "__main__":
    TEST_DIR = r"E:\DA_DeepLearning\Data\test\images" 
    MODEL_PATH = "models/faster_rcnn_accident.pth"
    num_classes = 4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()
    else:
        print("❌ Không tìm thấy file pth!")
        exit()

    list_imgs = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"📂 Đang quét {len(list_imgs)} ảnh. Nhấn phím bất kỳ để tiếp tục, 'Q' hoặc 'Esc' để thoát.")

    for img_name in list_imgs:
        result_img = test_inference(os.path.join(TEST_DIR, img_name), model, device)
        if result_img is not None:
            cv2.imshow("Demo Nhan dien Tai nan", result_img)
            
            # Đợi phím bấm
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27: # Thoát khi nhấn Q hoặc Esc
                break
    
    cv2.destroyAllWindows()