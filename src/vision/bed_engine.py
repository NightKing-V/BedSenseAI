"""
Dynamic Bed Calibration & 2D Polygon Overlap Engine for BedSense AI (§3.2 of Design Spec).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO


class BedRelationEngine:
    """
    Detects patient bed coordinates dynamically using YOLO in 2D image space
    and computes continuous geometric overlap without synthetic fallbacks or 3D homography assumptions.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        polygon: Optional[Union[List, np.ndarray]] = None,
        confidence_thresh: float = 0.12,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh
        self.device = device
        self.polygon: Optional[np.ndarray] = None
        self.bbox: Optional[Tuple[float, float, float, float]] = None
        self.is_calibrated: bool = False

        if polygon is not None:
            self.set_polygon(polygon)

    def auto_calibrate_from_video(self, video_path: Union[str, Path], sample_frames: int = 15) -> Optional[np.ndarray]:
        """Dynamically detect bed box from sample video frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self.polygon

        frames: List[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max(1, sample_frames))

        idx = 0
        while len(frames) < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()

        return self.calibrate_from_frames(frames)

    def calibrate_from_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Run YOLO to find median bed/couch box across sampled frames."""
        if not frames:
            self.polygon = None
            self.bbox = None
            self.is_calibrated = False
            return None

        boxes: List[Tuple[float, float, float, float]] = []
        try:
            model = YOLO(self.model_path)
            for frame in frames:
                res = model(frame, conf=self.confidence_thresh, verbose=False, device=self.device)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    for c, xyxy in zip(res.boxes.cls.cpu().numpy(), res.boxes.xyxy.cpu().numpy()):
                        if model.names.get(int(c)) in ["bed", "couch"]:
                            boxes.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])))
        except Exception:
            pass

        if boxes:
            x1, y1, x2, y2 = np.median(np.array(boxes), axis=0)
            self.set_polygon(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
            self.is_calibrated = True
        else:
            self.polygon = None
            self.bbox = None
            self.is_calibrated = False

        return self.polygon

    def set_polygon(self, points: Union[List, np.ndarray]):
        """Set 2D bed polygon coordinates in image space."""
        self.polygon = np.array(points, dtype=np.float32).reshape(-1, 2)
        if len(self.polygon) >= 3:
            self.bbox = (
                float(np.min(self.polygon[:, 0])),
                float(np.min(self.polygon[:, 1])),
                float(np.max(self.polygon[:, 0])),
                float(np.max(self.polygon[:, 1])),
            )
            self.is_calibrated = True
        else:
            self.polygon = None
            self.bbox = None
            self.is_calibrated = False

    def get_bed_longitudinal_axis(self) -> Optional[np.ndarray]:
        """Returns normalized 2D vector along the bed length axis in image space."""
        if self.polygon is None or len(self.polygon) < 4:
            return None
        top_mid = (self.polygon[0] + self.polygon[1]) / 2.0
        bot_mid = (self.polygon[3] + self.polygon[2]) / 2.0
        vec = bot_mid - top_mid
        norm = float(np.linalg.norm(vec))
        if norm > 1e-3:
            return vec / norm
        return None

    def get_bed_transverse_axis(self) -> Optional[np.ndarray]:
        """Returns normalized 2D vector along the bed width axis in image space."""
        if self.polygon is None or len(self.polygon) < 4:
            return None
        left_mid = (self.polygon[0] + self.polygon[3]) / 2.0
        right_mid = (self.polygon[1] + self.polygon[2]) / 2.0
        vec = right_mid - left_mid
        norm = float(np.linalg.norm(vec))
        if norm > 1e-3:
            return vec / norm
        return None

    def compute_overlap(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        keypoints: Optional[np.ndarray] = None,
    ) -> float:
        """
        Computes continuous geometric overlap ratio: Area(Person ∩ Bed) / Area(Person).
        Returns 0.0 if bed is not calibrated/detected.
        """
        if self.bbox is None or bbox is None or not self.is_calibrated:
            return 0.0

        px1, py1, px2, py2 = bbox
        bx1, by1, bx2, by2 = self.bbox

        p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        if p_area <= 1e-6:
            return 0.0

        ix1, iy1 = max(px1, bx1), max(py1, by1)
        ix2, iy2 = min(px2, bx2), min(py2, by2)
        i_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        area_overlap = i_area / p_area

        if keypoints is not None and len(keypoints) == 17:
            valid = keypoints[:, 2] > 0.25
            if np.any(valid):
                pts = keypoints[valid, :2]
                in_bed = np.sum((pts[:, 0] >= bx1) & (pts[:, 0] <= bx2) & (pts[:, 1] >= by1) & (pts[:, 1] <= by2))
                kpt_ratio = in_bed / len(pts)
                return float(np.clip(0.55 * area_overlap + 0.45 * kpt_ratio, 0.0, 1.0))

        return float(np.clip(area_overlap, 0.0, 1.0))

    def get_polygon(self) -> Optional[np.ndarray]:
        return self.polygon


__all__ = ["BedRelationEngine"]
