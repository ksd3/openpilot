"""Attention detector: head pose + eye aspect ratio (MediaPipe Face Mesh).

Answers: "Is at least one person in the frame looking at the robot?"
"""

import cv2
import numpy as np
import mediapipe as mp

from scp173.config import EAR_THRESHOLD, YAW_THRESHOLD_DEG, MAX_FACES


class AttentionDetector:
    """Determines if any person is actively watching the robot.

    Uses two signals:
      1. Eye Aspect Ratio (EAR) — open eyes required
      2. Head yaw via solvePnP — face must be oriented toward camera (|yaw| < threshold)
    """

    # MediaPipe face mesh landmark indices for eye contours
    _LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    _RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Landmark indices used for head pose (nose tip, left eye, right eye,
    # left mouth corner, right mouth corner, chin)
    _POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

    # Generic 3-D face model points (mm)
    _MODEL_3D = np.array([
        (  0.0,    0.0,    0.0),   # Nose tip
        (-225.0,  170.0, -135.0),  # Left eye corner
        ( 225.0,  170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        ( 150.0, -150.0, -125.0),  # Right mouth corner
        (  0.0,  -330.0,  -65.0),  # Chin
    ], dtype=np.float64)

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------
    def is_being_watched(self, frame: np.ndarray) -> tuple[bool, int]:
        """Return (watched: bool, num_faces_detected: int).

        watched=True means at least one person has open eyes and is facing
        the camera within YAW_THRESHOLD_DEG.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # No face visible at all — back is turned or person is absent
            return False, 0

        num_faces = len(results.multi_face_landmarks)
        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark

            # --- Head yaw first — if clearly turned away, skip EAR -------
            try:
                yaw = self._head_yaw(lm, frame.shape)
            except Exception:
                continue  # solvePnP failed — treat as not watching
            if abs(yaw) >= YAW_THRESHOLD_DEG:
                continue  # turned away

            # --- Eye aspect ratio — open eyes required --------------------
            left_ear  = self._ear(lm, self._LEFT_EYE)
            right_ear = self._ear(lm, self._RIGHT_EYE)
            if (left_ear + right_ear) / 2.0 < EAR_THRESHOLD:
                continue  # eyes closed → not watching

            # Facing camera with open eyes = WATCHING
            return True, num_faces

        return False, num_faces

    # ------------------------------------------------------------------
    def _ear(self, landmarks, indices: list[int]) -> float:
        """Eye Aspect Ratio: (v1 + v2) / (2 * h).  Low → closed."""
        pts = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h  = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _head_yaw(self, landmarks, frame_shape) -> float:
        """Estimate head yaw in degrees using solvePnP.
        Returns yaw; positive = rotated right, negative = rotated left.
        """
        h, w = frame_shape[:2]
        image_pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in self._POSE_LANDMARKS],
            dtype=np.float64,
        )
        focal = float(w)
        cam_matrix = np.array(
            [[focal, 0, w / 2],
             [0, focal, h / 2],
             [0, 0,       1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        _, rvec, _ = cv2.solvePnP(
            self._MODEL_3D, image_pts, cam_matrix, dist_coeffs
        )
        rmat, _ = cv2.Rodrigues(rvec)
        yaw = np.degrees(np.arctan2(rmat[0, 2], rmat[2, 2]))
        return float(yaw)
