from pathlib import Path

# resolve everything from the project root, not the current working directory
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
RAW_DIR = ROOT / "data" / "raw"  # source videos, never modified
DATA_DIR = ROOT / "data" / "processed"  # face crops generated from RAW_DIR, safe to delete
LABELED_DIR = ROOT / "labeled_data"

MODEL_PATH = MODELS_DIR / "model.pth"
FACE_DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
