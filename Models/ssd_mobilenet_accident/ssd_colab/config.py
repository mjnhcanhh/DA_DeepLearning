import os

USE_COLAB = os.path.exists("/content")

if USE_COLAB:
    DATA_DIR = "/content/drive/MyDrive/ssd_dataset"
    MODEL_SAVE_PATH = "/content/drive/MyDrive/ssd_accident_best.pth"
    LAST_CHECKPOINT_PATH = "/content/drive/MyDrive/ssd_accident_last.pth"
    LOG_DIR = "/content/drive/MyDrive/ssd_logs"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../.."))

    DATA_DIR = os.path.join(PROJECT_DIR, "Data", "ssd_dataset")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "ssd_accident_best.pth")
    LAST_CHECKPOINT_PATH = os.path.join(BASE_DIR, "ssd_accident_last.pth")
    LOG_DIR = os.path.join(BASE_DIR, "ssd_logs")

TRAIN_IMAGES = os.path.join(DATA_DIR, "images", "train")
TRAIN_ANNOS  = os.path.join(DATA_DIR, "annotations", "train")

VAL_IMAGES   = os.path.join(DATA_DIR, "images", "val")
VAL_ANNOS    = os.path.join(DATA_DIR, "annotations", "val")

TEST_IMAGES  = os.path.join(DATA_DIR, "images", "test")
TEST_ANNOS   = os.path.join(DATA_DIR, "annotations", "test")

CLASS_NAMES = ["background", "accident"]
NUM_CLASSES = len(CLASS_NAMES)

NUM_EPOCHS = 30
BATCH_SIZE = 8 if USE_COLAB else 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 2 if USE_COLAB else 0
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320

print("===== CONFIG =====")
print("USE_COLAB:", USE_COLAB)
print("DATA_DIR:", DATA_DIR)
print("TRAIN_IMAGES:", TRAIN_IMAGES)
print("TRAIN_ANNOS:", TRAIN_ANNOS)
print("VAL_IMAGES:", VAL_IMAGES)
print("VAL_ANNOS:", VAL_ANNOS)
print("TEST_IMAGES:", TEST_IMAGES)
print("TEST_ANNOS:", TEST_ANNOS)
print("MODEL_SAVE_PATH:", MODEL_SAVE_PATH)
print("LAST_CHECKPOINT_PATH:", LAST_CHECKPOINT_PATH)
print("LOG_DIR:", LOG_DIR)
print("==================")