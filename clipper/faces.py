"""Face detection with OpenCV's YuNet; the 230 KB model is downloaded on first use."""

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
NO_FACES = np.empty((0, 15), np.float32)

_detector = None


def detect(frame: np.ndarray) -> np.ndarray:
    """Faces as rows of 15 numbers: box x, y, w, h; then (x, y) of the right eye, left eye,
    nose tip, right and left mouth corner; then the confidence."""
    global _detector
    if _detector is None:
        _detector = _load()
    if _detector is False:
        return NO_FACES
    h, w = frame.shape[:2]
    _detector.setInputSize((w, h))
    _, found = _detector.detect(frame)
    return found if found is not None else NO_FACES


def _load():
    model = MODEL_DIR / MODEL_URL.rsplit("/", 1)[1]
    if not model.exists():
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            tmp = model.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(model)
        except OSError as exc:
            print(f"    (face detection off: couldn't download the face model: {exc})")
            return False
    return cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7, 0.3, 20)
