"""MobileNet-SSD v2 person detector (OpenCV DNN — no onnxruntime needed).

Drop-in replacement for person_detector.py. ~3-5x faster on ARM CPU.
"""

import cv2
import numpy as np
import os

_HERE = os.path.dirname(os.path.dirname(__file__))
DEFAULT_PB = os.path.join(_HERE, "models", "mobilenet_ssd_v2.pb")
DEFAULT_PBTXT = os.path.join(_HERE, "models", "mobilenet_ssd_v2.pbtxt")

# COCO class 1 = person
PERSON_CLASS_ID = 1


class PersonDetector:
    """Detect people using MobileNet-SSD v2 via OpenCV DNN.

    Returns list of (x1, y1, x2, y2, confidence) in original frame coords.
    """

    def __init__(self, pb_path: str = DEFAULT_PB, pbtxt_path: str = DEFAULT_PBTXT,
                 conf_thresh: float = 0.5, input_size: int = 300):
        self.net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_thresh = conf_thresh
        self.input_size = input_size

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """Run detection on a BGR frame.

        Returns list of (x1, y1, x2, y2, conf) in original frame coords.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, size=(self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            confidence = float(detections[0, 0, i, 2])

            if class_id != PERSON_CLASS_ID or confidence < self.conf_thresh:
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            results.append((float(x1), float(y1), float(x2), float(y2), confidence))

        return results
