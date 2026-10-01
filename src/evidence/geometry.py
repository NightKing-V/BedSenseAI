"""
Pose Landmark & Geometric Feature Extraction for BedSense AI Evidence Engine.
Implements 3-point joint angles, normalized trunk proportions, and anatomical geometry.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass
class BodyLandmarks:
    """Anatomical landmarks and synthesized functional midpoints."""
    nose: Optional[np.ndarray] = None

    shoulder_center: Optional[np.ndarray] = None
    hip_center: Optional[np.ndarray] = None
    knee_center: Optional[np.ndarray] = None
    ankle_center: Optional[np.ndarray] = None

    left_shoulder: Optional[np.ndarray] = None
    right_shoulder: Optional[np.ndarray] = None

    left_elbow: Optional[np.ndarray] = None
    right_elbow: Optional[np.ndarray] = None

    left_wrist: Optional[np.ndarray] = None
    right_wrist: Optional[np.ndarray] = None

    left_hip: Optional[np.ndarray] = None
    right_hip: Optional[np.ndarray] = None

    left_knee: Optional[np.ndarray] = None
    right_knee: Optional[np.ndarray] = None

    left_ankle: Optional[np.ndarray] = None
    right_ankle: Optional[np.ndarray] = None


def point_distance(point_a: Optional[np.ndarray], point_b: Optional[np.ndarray]) -> Optional[float]:
    """Euclidean distance between two 2D points."""
    if point_a is None or point_b is None:
        return None
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def joint_angle(
    point_a: Optional[np.ndarray],
    point_b: Optional[np.ndarray],
    point_c: Optional[np.ndarray],
) -> Optional[float]:
    """
    Calculate planar angle ABC in degrees with vertex at joint B.
    """
    if point_a is None or point_b is None or point_c is None:
        return None

    vec_a = np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float)
    vec_c = np.asarray(point_c, dtype=float) - np.asarray(point_b, dtype=float)

    norm_a = np.linalg.norm(vec_a)
    norm_c = np.linalg.norm(vec_c)

    if norm_a <= 1e-6 or norm_c <= 1e-6:
        return None

    cosine = np.dot(vec_a, vec_c) / (norm_a * norm_c)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def vector_between(
    point_a: Optional[np.ndarray],
    point_b: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Directed vector from point A to point B."""
    if point_a is None or point_b is None:
        return None
    return np.asarray(point_b, dtype=float) - np.asarray(point_a, dtype=float)


def normalize_vector(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize a vector to unit length."""
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm <= 1e-6:
        return None
    return np.asarray(vec, dtype=float) / norm


class LandmarkExtractor:
    """Extracts BodyLandmarks from 17-point COCO keypoint array."""

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = min_confidence

    def extract(self, keypoints: Optional[np.ndarray]) -> BodyLandmarks:
        if keypoints is None or len(keypoints) < 17:
            return BodyLandmarks()

        kpts = np.asarray(keypoints)

        def get_kp(name: str) -> Optional[np.ndarray]:
            idx = COCO_KEYPOINTS[name]
            if len(kpts.shape) == 2 and kpts.shape[1] >= 3:
                x, y, conf = kpts[idx][:3]
                if conf >= self.min_confidence:
                    return np.array([float(x), float(y)], dtype=float)
            elif len(kpts.shape) == 2 and kpts.shape[1] == 2:
                return np.array([float(kpts[idx][0]), float(kpts[idx][1])], dtype=float)
            return None

        ls = get_kp("left_shoulder")
        rs = get_kp("right_shoulder")
        lh = get_kp("left_hip")
        rh = get_kp("right_hip")
        lk = get_kp("left_knee")
        rk = get_kp("right_knee")
        la = get_kp("left_ankle")
        ra = get_kp("right_ankle")

        le = get_kp("left_elbow")
        re = get_kp("right_elbow")
        lw = get_kp("left_wrist")
        rw = get_kp("right_wrist")
        nose = get_kp("nose")

        return BodyLandmarks(
            nose=nose,
            shoulder_center=self._center(ls, rs),
            hip_center=self._center(lh, rh),
            knee_center=self._center(lk, rk),
            ankle_center=self._center(la, ra),
            left_shoulder=ls,
            right_shoulder=rs,
            left_hip=lh,
            right_hip=rh,
            left_knee=lk,
            right_knee=rk,
            left_ankle=la,
            right_ankle=ra,
            left_elbow=le,
            right_elbow=re,
            left_wrist=lw,
            right_wrist=rw,
        )

    @staticmethod
    def _center(pt_a: Optional[np.ndarray], pt_b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pt_a is not None and pt_b is not None:
            return (np.asarray(pt_a, dtype=float) + np.asarray(pt_b, dtype=float)) / 2.0
        if pt_a is not None:
            return np.asarray(pt_a, dtype=float).copy()
        if pt_b is not None:
            return np.asarray(pt_b, dtype=float).copy()
        return None


@dataclass
class PoseFeatures:
    """Extracted geometric and proportional pose features."""
    track_id: Optional[int] = None

    shoulder_width: Optional[float] = None
    torso_length: Optional[float] = None
    hip_to_knee: Optional[float] = None
    knee_to_ankle: Optional[float] = None

    # Invariant normalized ratios (scaled by shoulder width)
    torso_length_normalized: Optional[float] = None
    hip_to_knee_normalized: Optional[float] = None
    knee_to_ankle_normalized: Optional[float] = None

    # Biomechanical joint angles (degrees)
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    mean_knee_angle: Optional[float] = None

    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    mean_hip_angle: Optional[float] = None

    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None

    torso_direction_x: Optional[float] = None
    torso_direction_y: Optional[float] = None

    body_axis_angle: Optional[float] = None
    body_axis_length: Optional[float] = None


class PoseFeatureExtractor:
    """Computes joint angles and scale-invariant proportional features."""

    def extract(self, landmarks: BodyLandmarks, track_id: Optional[int] = None) -> PoseFeatures:
        shoulder_width = point_distance(landmarks.left_shoulder, landmarks.right_shoulder)
        torso_length = point_distance(landmarks.shoulder_center, landmarks.hip_center)
        hip_to_knee = point_distance(landmarks.hip_center, landmarks.knee_center)
        knee_to_ankle = point_distance(landmarks.knee_center, landmarks.ankle_center)

        left_knee_angle = joint_angle(landmarks.left_hip, landmarks.left_knee, landmarks.left_ankle)
        right_knee_angle = joint_angle(landmarks.right_hip, landmarks.right_knee, landmarks.right_ankle)
        mean_knee = self._average(left_knee_angle, right_knee_angle)

        left_hip_angle = joint_angle(landmarks.shoulder_center, landmarks.left_hip, landmarks.left_knee)
        right_hip_angle = joint_angle(landmarks.shoulder_center, landmarks.right_hip, landmarks.right_knee)
        mean_hip = self._average(left_hip_angle, right_hip_angle)

        left_elbow_angle = joint_angle(landmarks.left_shoulder, landmarks.left_elbow, landmarks.left_wrist)
        right_elbow_angle = joint_angle(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist)

        torso_vec = vector_between(landmarks.shoulder_center, landmarks.hip_center)
        torso_dir = normalize_vector(torso_vec)
        torso_dx = float(torso_dir[0]) if torso_dir is not None else None
        torso_dy = float(torso_dir[1]) if torso_dir is not None else None

        body_axis_angle = None
        body_axis_length = None
        if landmarks.shoulder_center is not None and landmarks.hip_center is not None:
            dx = landmarks.hip_center[0] - landmarks.shoulder_center[0]
            dy = landmarks.hip_center[1] - landmarks.shoulder_center[1]
            body_axis_length = float(np.sqrt(dx**2 + dy**2))
            body_axis_angle = float(np.degrees(np.arctan2(dy, dx)))

        return PoseFeatures(
            track_id=track_id,
            shoulder_width=shoulder_width,
            torso_length=torso_length,
            hip_to_knee=hip_to_knee,
            knee_to_ankle=knee_to_ankle,
            torso_length_normalized=self._normalize(torso_length, shoulder_width),
            hip_to_knee_normalized=self._normalize(hip_to_knee, shoulder_width),
            knee_to_ankle_normalized=self._normalize(knee_to_ankle, shoulder_width),
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            mean_knee_angle=mean_knee,
            left_hip_angle=left_hip_angle,
            right_hip_angle=right_hip_angle,
            mean_hip_angle=mean_hip,
            left_elbow_angle=left_elbow_angle,
            right_elbow_angle=right_elbow_angle,
            torso_direction_x=torso_dx,
            torso_direction_y=torso_dy,
            body_axis_angle=body_axis_angle,
            body_axis_length=body_axis_length,
        )

    @staticmethod
    def _normalize(value: Optional[float], scale: Optional[float]) -> Optional[float]:
        if value is None or scale is None or scale <= 1e-6:
            return None
        return float(value / scale)

    @staticmethod
    def _average(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is not None and b is not None:
            return float((a + b) / 2.0)
        if a is not None:
            return float(a)
        if b is not None:
            return float(b)
        return None
