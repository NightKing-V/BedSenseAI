"""
Vision Layer for BedSense AI (derived from src-old perception engine).
Handles YOLO-based dynamic bed calibration, single-pass YOLO-Pose detection, ByteTrack tracking,
robust multi-person resident tracking, and continuous geometric bed overlap computation in image space.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

from .contracts import FrameObservation


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
            # Explicitly uncalibrated when no bed detected (no synthetic fallback)
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


class PerceptionDetector:
    """
    Unified Primary Vision module.
    Runs YOLO-Pose with ByteTrack to track persons, extract 17 COCO keypoints,
    and maintains persistent resident identity when multiple people (caregivers, visitors) are present.
    """

    def __init__(
        self,
        model_path: Union[str, YOLO] = "yolo11n-pose.pt",
        conf_threshold: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
        device: str = "cuda:0",
        bed_engine: Optional[BedRelationEngine] = None,
        max_reacquire_dist: float = 180.0,
    ):
        if isinstance(model_path, str):
            self.model = YOLO(model_path)
        else:
            self.model = model_path

        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.device = device
        self.bed_engine = bed_engine or BedRelationEngine(device=device)
        self.max_reacquire_dist = max_reacquire_dist

        # Persistent Resident State
        self.primary_track_id: Optional[int] = None
        self.last_known_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_known_center: Optional[np.ndarray] = None
        self.last_known_kpts: Optional[np.ndarray] = None
        self.last_seen_ts: float = 0.0
        self.all_active_tracks: List[int] = []

    def auto_calibrate(self, video_source_or_path, sample_frames: int = 15):
        """Dynamically detect and calibrate bed area from video frames."""
        if hasattr(video_source_or_path, "video_path"):
            path = video_source_or_path.video_path
            self.bed_engine.auto_calibrate_from_video(path, sample_frames=sample_frames)
        elif isinstance(video_source_or_path, (str, Path)):
            self.bed_engine.auto_calibrate_from_video(str(video_source_or_path), sample_frames=sample_frames)

    def reset_tracker(self):
        """Reset ByteTrack tracker state."""
        try:
            from ultralytics.trackers.basetrack import BaseTrack
            BaseTrack.reset_id()
        except Exception:
            pass

        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                for t in self.model.predictor.trackers:
                    try:
                        t.reset()
                    except Exception:
                        pass
                try:
                    delattr(self.model.predictor, "trackers")
                except Exception:
                    pass
        self.primary_track_id = None
        self.last_known_bbox = None
        self.last_known_center = None
        self.last_known_kpts = None
        self.last_seen_ts = 0.0
        self.all_active_tracks = []

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> List[FrameObservation]:
        """
        Process single video frame with YOLO-Pose and ByteTrack.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        observations: List[FrameObservation] = []
        self.all_active_tracks = []

        if result.boxes is None or len(result.boxes) == 0:
            observations.append(
                FrameObservation(
                    t=timestamp,
                    track_id=None,
                    bbox=None,
                    keypoints=None,
                    bed_overlap=0.0,
                    mean_kpt_conf=0.0,
                )
            )
            return observations

        boxes = result.boxes
        has_keypoints = result.keypoints is not None

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes))

        if has_keypoints:
            kpt_xy = result.keypoints.xy.cpu().numpy()     # (N, 17, 2)
            kpt_conf = result.keypoints.conf.cpu().numpy() # (N, 17)
        else:
            kpt_xy = None
            kpt_conf = None

        for i in range(len(boxes)):
            tid = int(track_ids[i])
            self.all_active_tracks.append(tid)
            bbox: Tuple[float, float, float, float] = (
                float(xyxy[i, 0]),
                float(xyxy[i, 1]),
                float(xyxy[i, 2]),
                float(xyxy[i, 3]),
            )

            if has_keypoints and kpt_xy is not None and kpt_conf is not None:
                kpts_17_3 = np.zeros((17, 3), dtype=np.float32)
                kpts_17_3[:, :2] = kpt_xy[i]
                kpts_17_3[:, 2] = kpt_conf[i]
                mean_conf = float(np.mean(kpt_conf[i]))
            else:
                kpts_17_3 = None
                mean_conf = float(conf[i])

            overlap = self.bed_engine.compute_overlap(bbox, kpts_17_3)

            obs = FrameObservation(
                t=timestamp,
                track_id=tid,
                bbox=bbox,
                keypoints=kpts_17_3,
                bed_overlap=overlap,
                mean_kpt_conf=mean_conf,
            )
            observations.append(obs)

        return observations

    def get_primary_observation(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> FrameObservation:
        """
        Extracts observation for the primary resident with multi-person identity persistence,
        proximity-based reacquisition, and keypoint jump smoothing.
        """
        all_obs = self.process_frame(frame, timestamp)
        if not all_obs or (len(all_obs) == 1 and all_obs[0].track_id is None):
            return FrameObservation(
                t=timestamp,
                track_id=None,
                bbox=None,
                keypoints=None,
                bed_overlap=0.0,
                mean_kpt_conf=0.0,
            )

        # 1. First-time resident track initialization
        if self.primary_track_id is None:
            # Prefer person with highest bed overlap (resident in/on bed)
            sorted_obs = sorted(all_obs, key=lambda o: (o.bed_overlap, o.mean_kpt_conf), reverse=True)
            chosen = sorted_obs[0]
            self._update_resident_state(chosen, timestamp)
            return chosen

        # 2. Check if primary track ID exists in current frame
        direct_match = next((o for o in all_obs if o.track_id == self.primary_track_id), None)
        if direct_match is not None and direct_match.bbox is not None:
            # Check for sudden keypoint teleportation (> 250px jump in < 0.2s)
            c_new = np.array([(direct_match.bbox[0] + direct_match.bbox[2]) / 2.0,
                              (direct_match.bbox[1] + direct_match.bbox[3]) / 2.0])
            if self.last_known_center is not None:
                dist = float(np.linalg.norm(c_new - self.last_known_center))
                dt = max(1e-3, timestamp - self.last_seen_ts)
                if dist > 250.0 and dt < 0.3 and len(all_obs) > 1:
                    # Potential tracker swap to another person; verify if another track is closer
                    closest_candidate = self._find_closest_candidate(all_obs)
                    if closest_candidate is not None:
                        direct_match = closest_candidate

            smoothed_obs = self._smooth_keypoints(direct_match)
            self._update_resident_state(smoothed_obs, timestamp)
            return smoothed_obs

        # 3. Track reacquisition: ID dropped/changed in multi-person scene
        closest_candidate = self._find_closest_candidate(all_obs)
        if closest_candidate is not None:
            self.primary_track_id = closest_candidate.track_id
            smoothed_obs = self._smooth_keypoints(closest_candidate)
            self._update_resident_state(smoothed_obs, timestamp)
            return smoothed_obs

        # 4. Resident not detected near last known location (e.g. only caregiver elsewhere in room)
        # Avoid switching to caregiver
        return FrameObservation(
            t=timestamp,
            track_id=None,
            bbox=None,
            keypoints=None,
            bed_overlap=0.0,
            mean_kpt_conf=0.0,
        )

    def _find_closest_candidate(self, observations: List[FrameObservation]) -> Optional[FrameObservation]:
        """Finds candidate track matching resident's spatial proximity and bed overlap."""
        if not observations or self.last_known_center is None or self.last_known_bbox is None:
            return None

        candidates = [o for o in observations if o.bbox is not None]
        if not candidates:
            return None

        best_obs: Optional[FrameObservation] = None
        min_dist = float("inf")

        for obs in candidates:
            bx1, by1, bx2, by2 = obs.bbox  # type: ignore
            c = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            dist = float(np.linalg.norm(c - self.last_known_center))

            # Calculate BBox IoU
            iou = self._bbox_iou(obs.bbox, self.last_known_bbox)  # type: ignore

            # If inside bed region or close to last location
            if iou > 0.15 or dist < self.max_reacquire_dist:
                if dist < min_dist:
                    min_dist = dist
                    best_obs = obs

        return best_obs

    def _update_resident_state(self, obs: FrameObservation, timestamp: float):
        """Updates internal spatial tracker state."""
        self.primary_track_id = obs.track_id
        self.last_seen_ts = timestamp
        if obs.bbox is not None:
            self.last_known_bbox = obs.bbox
            self.last_known_center = np.array([
                (obs.bbox[0] + obs.bbox[2]) / 2.0,
                (obs.bbox[1] + obs.bbox[3]) / 2.0,
            ])
        if obs.keypoints is not None:
            self.last_known_kpts = obs.keypoints.copy()

    def _smooth_keypoints(self, obs: FrameObservation) -> FrameObservation:
        """Applies EMA smoothing to keypoints to eliminate single-frame coordinate jitter."""
        if obs.keypoints is None or self.last_known_kpts is None:
            return obs

        kpts = obs.keypoints.copy()
        for j in range(17):
            if kpts[j, 2] > 0.25 and self.last_known_kpts[j, 2] > 0.25:
                # 85% new measurement + 15% previous frame
                kpts[j, :2] = 0.85 * kpts[j, :2] + 0.15 * self.last_known_kpts[j, :2]

        return FrameObservation(
            t=obs.t,
            track_id=obs.track_id,
            bbox=obs.bbox,
            keypoints=kpts,
            bed_overlap=obs.bed_overlap,
            mean_kpt_conf=obs.mean_kpt_conf,
        )

    @staticmethod
    def _bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        """Calculates 2D Intersection-over-Union."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return float(inter / max(1e-6, union))
