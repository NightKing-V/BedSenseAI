"""
Posture Geometry Frame Classifier for BedSense AI (§3 & §8 of Design Spec).
Maps posture geometry × bed overlap into the 7 canonical states:
LYING_IN_BED | SITTING_ON_BED | SITTING_OUTSIDE_BED | STANDING | WALKING | OUT_OF_BED | UNKNOWN
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..constants import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_UNKNOWN,
    STATE_WALKING,
)
from ..contracts import FrameObservation


class FrameClassifier:
    """
    Posture Geometry Frame Classifier (§3 & §8 of Design Spec).
    """

    def __init__(
        self,
        in_bed_thresh: float = 0.50,
        out_of_bed_thresh: float = 0.10,
        min_kpt_conf_unknown: float = 0.25,
        torso_angle_lying_thresh: float = 50.0,
        torso_angle_upright_max: float = 35.0,
        aspect_ratio_standing_min: float = 1.35,
        aspect_ratio_lying_max: float = 0.95,
        knee_angle_bent_max: float = 135.0,
        walking_velocity_thresh: float = 25.0,
    ):
        self.in_bed_thresh = in_bed_thresh
        self.out_of_bed_thresh = out_of_bed_thresh
        self.min_kpt_conf_unknown = min_kpt_conf_unknown
        self.torso_angle_lying_thresh = torso_angle_lying_thresh
        self.torso_angle_upright_max = torso_angle_upright_max
        self.aspect_ratio_standing_min = aspect_ratio_standing_min
        self.aspect_ratio_lying_max = aspect_ratio_lying_max
        self.knee_angle_bent_max = knee_angle_bent_max
        self.walking_velocity_thresh = walking_velocity_thresh

        self.prev_center: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None
        self.was_out_of_bed: bool = False
        self.last_metrics: Dict[str, Any] = {}

    def reset(self):
        self.prev_center = None
        self.prev_t = None
        self.was_out_of_bed = False
        self.last_metrics = {}

    def get_last_metrics(self) -> Dict[str, Any]:
        """Returns biomechanical and temporal metrics calculated during the last frame classification."""
        return dict(self.last_metrics)

    def classify(
        self,
        obs: FrameObservation,
        bed_engine: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """Classifies raw observation into (canonical state, confidence) relative to bed plane."""
        if obs.track_id is None or obs.bbox is None or obs.keypoints is None:
            self.last_metrics = {
                "torso_angle": None,
                "knee_angle": None,
                "aspect_ratio": None,
                "velocity": 0.0,
                "hip_center": None,
                "is_aligned_with_bed_plane": False,
            }
            return STATE_UNKNOWN, 0.35

        if obs.mean_kpt_conf < self.min_kpt_conf_unknown:
            self.last_metrics = {
                "torso_angle": None,
                "knee_angle": None,
                "aspect_ratio": None,
                "velocity": 0.0,
                "hip_center": None,
                "is_aligned_with_bed_plane": False,
            }
            return STATE_UNKNOWN, float(obs.mean_kpt_conf)

        kpts = obs.keypoints
        x1, y1, x2, y2 = obs.bbox
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        aspect_ratio = bbox_h / bbox_w

        # 1. Joint Centers
        sh_pt = self._joint_center(kpts, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_pt = self._joint_center(kpts, LEFT_HIP, RIGHT_HIP)
        l_knee = kpts[LEFT_KNEE, :2] if kpts[LEFT_KNEE, 2] > 0.25 else None
        r_knee = kpts[RIGHT_KNEE, :2] if kpts[RIGHT_KNEE, 2] > 0.25 else None
        l_ankle = kpts[LEFT_ANKLE, :2] if kpts[LEFT_ANKLE, 2] > 0.25 else None
        r_ankle = kpts[RIGHT_ANKLE, :2] if kpts[RIGHT_ANKLE, 2] > 0.25 else None
        l_hip = kpts[LEFT_HIP, :2] if kpts[LEFT_HIP, 2] > 0.25 else None
        r_hip = kpts[RIGHT_HIP, :2] if kpts[RIGHT_HIP, 2] > 0.25 else None

        # 2. Torso Vector & Angles
        torso_angle_vert = 0.0
        has_torso = False
        unit_torso = None

        if sh_pt is not None and hip_pt is not None:
            dx, dy = sh_pt[0] - hip_pt[0], sh_pt[1] - hip_pt[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 1e-3:
                unit_torso = np.array([dx / length, dy / length])
                cos_val = np.clip(abs(dy) / length, -1.0, 1.0)
                torso_angle_vert = float(np.degrees(np.arccos(cos_val)))
                has_torso = True

        # 3. Torso Angle Relative to Bed Longitudinal Axis
        is_aligned_with_bed_plane = False

        if bed_engine is not None and getattr(bed_engine, "is_calibrated", False):
            bed_axis = bed_engine.get_bed_longitudinal_axis()
            if bed_axis is not None and unit_torso is not None:
                dot_bed = abs(float(np.dot(unit_torso, bed_axis)))
                if dot_bed >= 0.65 and obs.bed_overlap >= 0.30:
                    is_aligned_with_bed_plane = True

        # 4. Knee Angles (bent vs straight)
        knee_angles = []
        if l_hip is not None and l_knee is not None and l_ankle is not None:
            ang_l = self._joint_angle(l_hip, l_knee, l_ankle)
            if ang_l is not None:
                knee_angles.append(ang_l)
        if r_hip is not None and r_knee is not None and r_ankle is not None:
            ang_r = self._joint_angle(r_hip, r_knee, r_ankle)
            if ang_r is not None:
                knee_angles.append(ang_r)
        mean_knee_angle = float(np.mean(knee_angles)) if knee_angles else None

        # 5. Velocity
        center = hip_pt if hip_pt is not None else np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        velocity = 0.0
        if self.prev_center is not None and self.prev_t is not None:
            dt = obs.t - self.prev_t
            if 0 < dt < 1.0:
                velocity = float(np.linalg.norm(center - self.prev_center) / dt)

        self.prev_center = center.copy()
        self.prev_t = obs.t

        # Record calculated posture biomechanical metrics for telemetry export
        self.last_metrics = {
            "torso_angle": round(torso_angle_vert, 2) if has_torso else None,
            "knee_angle": round(mean_knee_angle, 2) if mean_knee_angle is not None else None,
            "aspect_ratio": round(aspect_ratio, 3),
            "velocity": round(velocity, 2),
            "hip_center": [round(float(c), 1) for c in center] if center is not None else None,
            "is_aligned_with_bed_plane": bool(is_aligned_with_bed_plane),
        }

        # 6. Simplified Posture & Bed Region Rule Tree (matching script.py)
        overlap = obs.bed_overlap
        in_bed = overlap >= 0.35

        # Horizontal / Lying: Torso angle >= 45° or wide aspect ratio with non-vertical torso
        is_horizontal = (has_torso and torso_angle_vert >= self.torso_angle_lying_thresh) or (
            aspect_ratio < self.aspect_ratio_lying_max and (not has_torso or torso_angle_vert > 40.0)
        )

        # Walking: Active translation displacement velocity
        is_walking = velocity >= self.walking_velocity_thresh

        # Standing: Tall vertical aspect ratio (H/W >= 1.35) or upright vertical torso (<= 35°) without bent sitting knees
        is_standing = (aspect_ratio >= self.aspect_ratio_standing_min or (has_torso and torso_angle_vert <= self.torso_angle_upright_max)) and (
            mean_knee_angle is None or mean_knee_angle >= self.knee_angle_bent_max or aspect_ratio >= 1.60
        )

        # 7. Map to Canonical 7 States via Bed Region
        if in_bed:
            if is_walking:
                state = STATE_WALKING
                conf = 0.90
            elif is_horizontal:
                state = STATE_LYING_IN_BED
                conf = 0.92 if overlap >= self.in_bed_thresh else 0.70
            else:
                state = STATE_SITTING_ON_BED
                conf = 0.90 if overlap >= self.in_bed_thresh else 0.70
        else:
            if is_walking:
                state = STATE_WALKING
                conf = 0.90
            elif is_horizontal:
                state = STATE_OUT_OF_BED
                conf = 0.85
            elif is_standing:
                state = STATE_STANDING
                conf = 0.90
            else:
                state = STATE_SITTING_OUTSIDE_BED
                conf = 0.85

        if overlap <= self.out_of_bed_thresh:
            self.was_out_of_bed = True
        elif overlap >= self.in_bed_thresh and state in (STATE_LYING_IN_BED, STATE_SITTING_ON_BED):
            self.was_out_of_bed = False

        final_conf = float(np.clip(conf * (0.5 + 0.5 * obs.mean_kpt_conf), 0.1, 0.99))
        return state, final_conf

    @staticmethod
    def _joint_center(kpts: np.ndarray, idx_a: int, idx_b: int) -> Optional[np.ndarray]:
        pt_a, conf_a = kpts[idx_a, :2], kpts[idx_a, 2]
        pt_b, conf_b = kpts[idx_b, :2], kpts[idx_b, 2]
        if conf_a > 0.2 and conf_b > 0.2:
            return (pt_a + pt_b) / 2.0
        elif conf_a > 0.25:
            return pt_a
        elif conf_b > 0.25:
            return pt_b
        return None

    @staticmethod
    def _joint_angle(pt_a: Optional[np.ndarray], pt_b: Optional[np.ndarray], pt_c: Optional[np.ndarray]) -> Optional[float]:
        """Computes angle at vertex B formed by rays BA and BC in degrees."""
        if pt_a is None or pt_b is None or pt_c is None:
            return None
        v1 = pt_a - pt_b
        v2 = pt_c - pt_b
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-3 or n2 < 1e-3:
            return None
        cos_val = np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))


__all__ = ["FrameClassifier"]
