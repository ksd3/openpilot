"""YOLOv8-nano person detector (ONNX Runtime)."""

import cv2
import numpy as np
import onnxruntime as ort

from scp173.config import YOLO_INPUT_SIZE, YOLO_CONF_THRESH


class PersonDetector:
    """Detect people in a BGR frame using YOLOv8-nano ONNX model.

    Returns a list of (x1, y1, x2, y2, confidence) tuples in original
    frame pixel coordinates, filtered to class 0 (person) only.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str, conf_thresh: float = YOLO_CONF_THRESH):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.conf_thresh = conf_thresh
        self.input_size = YOLO_INPUT_SIZE
        self._input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[tuple]:
        """Run detection on a BGR frame.

        Returns list of (x1, y1, x2, y2, conf) in original frame coords.
        """
        orig_h, orig_w = frame.shape[:2]
        blob = self._preprocess(frame)
        outputs = self.session.run(None, {self._input_name: blob})
        return self._postprocess(outputs[0], orig_w, orig_h)

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]  # NCHW

    def _postprocess(
        self, output: np.ndarray, orig_w: int, orig_h: int
    ) -> list[tuple]:
        """Parse YOLOv8 output tensor and return person boxes.

        YOLOv8 ONNX output shape: (1, 84, num_anchors) for COCO 80-class.
        Layout per anchor: [cx, cy, w, h, class0_score, class1_score, ...]
        """
        # output shape: (1, 84, N) → squeeze to (84, N) then transpose to (N, 84)
        preds = output[0].T  # (N, 84)

        # Extract box coords and class scores
        boxes = preds[:, :4]          # cx, cy, w, h (normalised to input size)
        scores = preds[:, 4:]         # class scores (no separate objectness in v8)
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        # Filter persons above threshold
        mask = (class_ids == self.PERSON_CLASS_ID) & (confidences >= self.conf_thresh)
        boxes = boxes[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2 (still in input-size coords)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Scale to original frame size
        sx = orig_w / self.input_size
        sy = orig_h / self.input_size
        x1 = np.clip(x1 * sx, 0, orig_w)
        y1 = np.clip(y1 * sy, 0, orig_h)
        x2 = np.clip(x2 * sx, 0, orig_w)
        y2 = np.clip(y2 * sy, 0, orig_h)

        # NMS
        rects = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        confs = confidences.tolist()
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(), confs, self.conf_thresh, nms_threshold=0.45
        )

        results = []
        for i in (indices.flatten() if len(indices) else []):
            results.append((
                float(x1[i]), float(y1[i]),
                float(x2[i]), float(y2[i]),
                float(confidences[i]),
            ))
        return results
