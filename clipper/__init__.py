"""Turn long narrated videos into vertical short-form clips."""

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # must be set before cv2 is imported
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Whisper model cache on Windows
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

__version__ = "0.1.0"
