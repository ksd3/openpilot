"""Monocular depth estimator — Depth Anything V2 Small (ONNX)."""

import cv2
import numpy as np
import onnxruntime as ort

from scp173.config import DEPTH_INPUT_SIZE

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator:
    """Estimate relative per-pixel depth from a single BGR frame.

    Output depth map shape: (DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE)
    Values normalised to [0, 1] where 0 = near, 1 = far.
    """

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self.session.get_inputs()[0].name
        self.input_size = DEPTH_INPUT_SIZE

    # ------------------------------------------------------------------
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return normalised depth map (H, W) where 0=near, 1=far."""
        blob = self._preprocess(frame)
        raw = self.session.run(None, {self._input_name: blob})[0][0]  # (H, W)
        depth = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return depth.astype(np.float32)

    def get_obstacle_map(
        self, depth: np.ndarray, near_threshold: float = 0.3
    ) -> np.ndarray:
        """Binary mask: True where an obstacle is too close.

        Ignores the bottom quarter of the frame (floor/ground plane).
        """
        h = depth.shape[0]
        roi = depth[:int(h * 0.75), :]
        return roi < near_threshold

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # NCHW
