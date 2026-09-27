from pathlib import Path

import cv2

FACE_SIZE = 128
PAD_RATIO = 0.2          # padding on each side, as a fraction of the face box
SCORE_THRESHOLD = 0.7    # min detection confidence (YuNet default is 0.9)
NMS_THRESHOLD = 0.3

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"

_detector = None


def _get_detector():
    """Load YuNet once, on first use."""
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {MODEL_PATH}. Download face_detection_yunet_2023mar.onnx "
                "from the opencv_zoo repo into ml/models/."
            )
        _detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (320, 320),
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
        )
    return _detector


def detect_face(frame):
    """Return the (x, y, w, h) int box of the largest face in a BGR frame, or None."""
    detector = _get_detector()
    frame_h, frame_w = frame.shape[:2]
    detector.setInputSize((frame_w, frame_h))  # must match the frame being passed in

    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None

    # each row: x, y, w, h, 5 landmark (x, y) pairs, score
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
    return int(x), int(y), int(w), int(h)


def crop_face_from_frame(frame, size: int = FACE_SIZE):
    """Crop a padded, square face region from a BGR frame and resize it.

    Returns a (size, size, 3) BGR array, or None if no face was found.
    """
    box = detect_face(frame)
    if box is None:
        return None

    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]

    # square box centred on the face, padded on every side
    side = int(max(w, h) * (1 + 2 * PAD_RATIO))
    side = min(side, frame_w, frame_h)
    if side <= 0:
        return None
    cx, cy = x + w // 2, y + h // 2

    # clamp to the frame while keeping the box square
    # (YuNet boxes can extend past the frame edge)
    x1 = min(max(0, cx - side // 2), frame_w - side)
    y1 = min(max(0, cy - side // 2), frame_h - side)
    crop = frame[y1:y1 + side, x1:x1 + side]

    # INTER_AREA is the right interpolation for downscaling
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)