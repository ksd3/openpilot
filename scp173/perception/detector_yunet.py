"""YuNet face detector via OpenCV FaceDetectorYN.

~31ms at 320x240 = 32fps. Detects faces at distance across full frame.
230KB ONNX model built into OpenCV.
"""

import cv2
import numpy as np


class YuNetDetector:
    """Combined person + attention detector using YuNet.

    detect() returns:
        faces: list of (x1, y1, x2, y2, conf) face boxes in original pixel coords
        being_watched: True if any face detected
        bearing: normalized bearing to largest face (-1..+1), or None
    """

    def __init__(self, model_path: str, input_size: tuple = (320, 240),
                 conf_threshold: float = 0.3, nms_threshold: float = 0.3):
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", input_size, conf_threshold, nms_threshold, 5000
        )
        self._input_size = input_size

    def detect(self, frame: np.ndarray) -> tuple[list, bool, float | None]:
        """Run face detection on a BGR or RGB frame.

        Returns:
            faces: list of (x1, y1, x2, y2, conf) in original frame coords
            being_watched: True if any face detected
            bearing: bearing to largest face (-1..+1), or None
        """
        h, w = frame.shape[:2]

        # Resize to input size for detection
        resized = cv2.resize(frame, self._input_size)
        self._detector.setInputSize(self._input_size)

        _, raw_faces = self._detector.detect(resized)

        faces = []
        if raw_faces is not None:
            sx = w / self._input_size[0]
            sy = h / self._input_size[1]
            for face in raw_faces:
                x1 = float(face[0] * sx)
                y1 = float(face[1] * sy)
                x2 = float((face[0] + face[2]) * sx)
                y2 = float((face[1] + face[3]) * sy)
                conf = float(face[-1])
                faces.append((x1, y1, x2, y2, conf))

        being_watched = len(faces) > 0
        bearing = None

        if faces:
            largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            cx = (largest[0] + largest[2]) / 2.0
            bearing = (cx / w) * 2.0 - 1.0

        return faces, being_watched, bearing
