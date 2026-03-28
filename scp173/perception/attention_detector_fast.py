"""Fast attention detector using OpenCV Haar cascade.

If a face is visible → person is watching. No face → not watching.
~20ms on ARM CPU vs ~200-300ms for MediaPipe face mesh.
"""

import cv2
import numpy as np


class AttentionDetector:
    """Determines if any person is actively watching the robot.

    Simple heuristic: if OpenCV detects a face, they're watching.
    No face detected = turned away or too far = not watching.
    """

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def is_being_watched(self, frame: np.ndarray) -> tuple[bool, int]:
        """Return (watched: bool, num_faces_detected: int)."""
        # Downscale for speed
        h, w = frame.shape[:2]
        scale = min(320 / w, 320 / h, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(30, 30),
        )

        num_faces = len(faces)
        return num_faces > 0, num_faces
