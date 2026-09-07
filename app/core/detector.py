"""YOLO-based person detection, including optional segmentation masks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    """The detector output needed by the tracker: location, confidence, and mask."""
    box: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray | None = None


class PersonDetector:
    """Wrap a YOLO model and expose only detections for COCO class 0 (person)."""
    def __init__(self, model_name: str, confidence: float = 0.35, device: str = ''):
        """Load the model once; ``confidence`` filters weak detections."""
        self.model = YOLO(model_name)
        self.confidence, self.device = confidence, device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO and convert its tensors into simple NumPy-friendly records."""
        result = self.model(
            frame,
            classes=[0],  # COCO class 0 is a person, so other objects are ignored.
            conf=self.confidence,
            iou=0.7,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        masks = None
        if result.masks is not None:
            masks_np = result.masks.data.cpu().numpy()
            h, w = frame.shape[:2]
            # YOLO may return masks at a smaller resolution than the original frame.
            if masks_np.shape[1:] != (h, w):
                masks_resized = []
                for m in masks_np:
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks_resized.append(m_resized)
                masks = np.array(masks_resized)
            else:
                masks = masks_np

        detections = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            mask = masks[i] if masks is not None else None
            detections.append(Detection(tuple(box), float(score), mask))
        return detections
