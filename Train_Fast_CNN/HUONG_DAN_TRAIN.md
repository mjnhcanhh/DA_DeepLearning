# Huong dan ket noi va train model

Tai lieu nay dung cho project `Train_Fast_CNN`.

## 1. Cau truc dung cua project

Sau khi giai nen, project nen co dang:

```text
Train_Fast_CNN/
  train.py
  code/
    dataset.py
    model.py
    utils.py
  Accident.v2i.voc/
    train/
    valid/
    test/
```

Neu ban van thay `code/train.py` tren Colab thi do la file zip cu. Khi chay, hay chay o thu muc goc:

```python
%cd /content/Train_Fast_CNN
!python train.py
```

Khong chay trong:

```python
%cd /content/Train_Fast_CNN/code
```

Vi neu chay trong `code`, chuong trinh se tim sai duong dan dataset.

## 2. Chay tren Colab GPU

### Buoc 1: Bat GPU

Trong Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU -> Save
```

Kiem tra GPU:

```python
import torch

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Ket qua dung nen la:

```text
CUDA: True
Tesla T4
```

### Buoc 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Buoc 3: Giai nen project

Neu ban upload file `Train_Fast_CNN.zip` vao MyDrive:

```python
!rm -rf /content/Train_Fast_CNN
!unzip -q /content/drive/MyDrive/Train_Fast_CNN.zip -d /content/
```

Kiem tra:

```python
!ls /content/Train_Fast_CNN
!ls /content/Train_Fast_CNN/code
!ls /content/Train_Fast_CNN/Accident.v2i.voc
```

Can thay:

```text
train.py
code/dataset.py
code/model.py
code/utils.py
Accident.v2i.voc/train
Accident.v2i.voc/valid
Accident.v2i.voc/test
```

### Buoc 4: Train

```python
%cd /content/Train_Fast_CNN
!python train.py
```

## 3. Noi luu ket qua tren Colab

Khi chay tren Colab va da mount Drive, ket qua se luu tai:

```text
/content/drive/MyDrive/HUIT_Project/outputs/
```

Ben trong co:

```text
outputs/
  weights/
    last_model.pth
    best_model.pth
    epoch_005.pth
    epoch_010.pth
    final_model.pth
  plots/
    training_dashboard_latest.png
    confusion_matrix_latest.png
  history.json
```

Y nghia:

```text
last_model.pth  : checkpoint moi nhat, dung de train tiep neu Colab bi tat
best_model.pth  : model co validation mAP tot nhat
epoch_XXX.pth   : checkpoint dinh ky
history.json    : loss, mAP, LR qua tung epoch
plots/          : anh so do training, confusion matrix, PR/F1 curve
```

## 4. Vi sao plots chua co anh?

Neu folder `plots` trong Drive dang trong, thuong la do chua toi luc ve bieu do.

Trong code co cau hinh:

```python
"plot_every": 2
```

Nghia la chi tao anh sau epoch:

```text
2, 4, 6, ...
```

Neu dang o epoch 1 thi `plots` trong la binh thuong.

Muon epoch 1 co anh luon thi sua trong `train.py`:

```python
"plot_every": 1
```

Sau khi sua, can chay lai tu dau hoac tu checkpoint moi.

## 5. Resume sau khi Colab timeout

Neu muon lan sau tu dong train tiep tu `last_model.pth`, sua trong `train.py`:

```python
"auto_resume": True
```

Neu muon train lai tu dau, de:

```python
"resume_from": None,
"auto_resume": False,
```

Va neu can xoa checkpoint cu tren Drive:

```python
!rm -f /content/drive/MyDrive/HUIT_Project/outputs/weights/last_model.pth
```

## 6. Chay local tren laptop RTX 3060

Mo terminal tai:

```powershell
E:\Train_Fast_CNN
```

Kiem tra CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Train:

```powershell
python train.py
```

Ket qua local luu tai:

```text
E:\Train_Fast_CNN\outputs\
```

## 7. Loi thuong gap

### Loi: Khong tim thay dataset

Vi du:

```text
Khong tim thay: /content/Train_Fast_CNN/code/Accident.v2i.voc/train
```

Nguyen nhan: ban dang chay trong thu muc `code`.

Sua:

```python
%cd /content/Train_Fast_CNN
!python train.py
```

### Loi: CUDA False

Nguyen nhan: chua bat GPU.

Sua:

```text
Runtime -> Change runtime type -> T4 GPU
```

### Canh bao ShiftScaleRotate

Neu thay:

```text
ShiftScaleRotate is a special case of Affine transform
```

Day chi la warning cua albumentations, khong phai loi. Train van chay duoc.

### Dang tai file `.pth` 167 MB

Day la pretrained weight cua Faster R-CNN tu PyTorch:

```text
fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
```

No duoc dung de fine-tune model, khong phai loi.

## 8. Lenh Colab day du nen dung

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

```python
!rm -rf /content/Train_Fast_CNN
!unzip -q /content/drive/MyDrive/Train_Fast_CNN.zip -d /content/
```

```python
%cd /content/Train_Fast_CNN
!python train.py
```

