"""BlazeFace-based person + attention detector.

Uses MediaPipe's BlazeFace (TFLite + XNNPACK) for everything:
- Face detected = person is there + watching → FREEZE
- Face disappears = turned away or left → MOVE toward last known position
- No face = nobody → IDLE

~24ms per frame = 40+ fps on comma 4 CPU.
"""

import mediapipe as mp
import numpy as np


class BlazeFaceDetector:
    """Combined person detector + attention detector using BlazeFace.

    detect() returns:
        people: list of (x1, y1, x2, y2, conf) face boxes in pixel coords
        being_watched: True if any face is detected (facing camera = watching)
        bearing: normalized bearing to largest face (-1 left, +1 right), or None
    """

    def __init__(self, model_path: str, confidence: float = 0.5):
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> tuple[list, bool, float | None]:
        """Run face detection on an RGB frame.

        Returns:
            faces: list of (x1, y1, x2, y2, conf) in pixel coords
            being_watched: True if any face detected
            bearing: bearing to largest face (-1..+1), or None if no face
        """
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)

        faces = []
        for det in result.detections:
            bb = det.bounding_box
            x1 = bb.origin_x
            y1 = bb.origin_y
            x2 = bb.origin_x + bb.width
            y2 = bb.origin_y + bb.height
            conf = det.categories[0].score if det.categories else 0.5
            faces.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        being_watched = len(faces) > 0
        bearing = None

        if faces:
            # Pick largest face
            largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            cx = (largest[0] + largest[2]) / 2.0
            bearing = (cx / w) * 2.0 - 1.0  # -1 left, +1 right

        return faces, being_watched, bearing
