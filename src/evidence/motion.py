"""
Multi-Joint Motion Sensing & Velocity Tracking for BedSense AI Evidence Engine.
Implements limb-specific kinematics (ankles, knees, hips, shoulders) and rolling profile smoothing.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple
import numpy as np

from .geometry import BodyLandmarks


@dataclass
class MotionVector:
    """Displacement and velocity between two consecutive frames."""
    dx: Optional[float] = None
    dy: Optional[float] = None
    magnitude: Optional[float] = None
    velocity: Optional[float] = None


def point_displacement(
    point_a: Optional[np.ndarray],
    point_b: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """2D displacement vector from point_a to point_b."""
    if point_a is None or point_b is None:
        return None
    return np.asarray(point_b, dtype=float) - np.asarray(point_a, dtype=float)


def calculate_motion_vector(
    previous_point: Optional[np.ndarray],
    current_point: Optional[np.ndarray],
    dt: Optional[float],
) -> MotionVector:
    """Calculates displacement and velocity between two consecutive temporal observations."""
    displacement = point_displacement(previous_point, current_point)
    if displacement is None:
        return MotionVector()

    magnitude = float(np.linalg.norm(displacement))
    velocity = magnitude / dt if dt is not None and dt > 0.0 else None

    return MotionVector(
        dx=float(displacement[0]),
        dy=float(displacement[1]),
        magnitude=magnitude,
        velocity=velocity,
    )


@dataclass
class BodyMotionProfile:
    """Smoothed limb and overall body velocity metrics."""
    timestamp: float

    shoulder_velocity: Optional[float] = None
    hip_velocity: Optional[float] = None
    knee_velocity: Optional[float] = None
    ankle_velocity: Optional[float] = None
    whole_body_motion: Optional[float] = None


@dataclass
class BodyCenterMotion:
    """Body center position and instantaneous velocity."""
    timestamp: float
    x: Optional[float] = None
    y: Optional[float] = None
    displacement: Optional[float] = None
    velocity: Optional[float] = None


class EvidenceMotionTracker:
    """
    Maintains per-track motion histories and calculates smoothed limb velocities.
    Operates in normalized coordinate space [0, 1] relative to frame dimensions.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

        self.prev_timestamp: Optional[float] = None
        self.prev_norm_landmarks: Optional[BodyLandmarks] = None
        self.prev_center: Optional[np.ndarray] = None

        self.shoulder_history: Deque[float] = deque(maxlen=window_size)
        self.hip_history: Deque[float] = deque(maxlen=window_size)
        self.knee_history: Deque[float] = deque(maxlen=window_size)
        self.ankle_history: Deque[float] = deque(maxlen=window_size)
        self.center_history: Deque[float] = deque(maxlen=window_size)

    def reset(self):
        self.prev_timestamp = None
        self.prev_norm_landmarks = None
        self.prev_center = None
        self.shoulder_history.clear()
        self.hip_history.clear()
        self.knee_history.clear()
        self.ankle_history.clear()
        self.center_history.clear()

    def update(
        self,
        timestamp: float,
        landmarks: BodyLandmarks,
        frame_width: float = 1280.0,
        frame_height: float = 720.0,
    ) -> Tuple[BodyMotionProfile, BodyCenterMotion]:
        """
        Updates motion state with new landmarks and returns profile & center motion.
        """
        # Normalize landmark positions to [0, 1] frame coordinates
        norm_landmarks = self._normalize_landmarks(landmarks, frame_width, frame_height)
        curr_center = norm_landmarks.hip_center if norm_landmarks.hip_center is not None else norm_landmarks.shoulder_center

        if self.prev_timestamp is None or self.prev_norm_landmarks is None:
            self.prev_timestamp = timestamp
            self.prev_norm_landmarks = norm_landmarks
            self.prev_center = curr_center

            return (
                BodyMotionProfile(timestamp=timestamp),
                BodyCenterMotion(
                    timestamp=timestamp,
                    x=float(curr_center[0]) if curr_center is not None else None,
                    y=float(curr_center[1]) if curr_center is not None else None,
                ),
            )

        dt = max(1e-4, timestamp - self.prev_timestamp)

        # 1. Individual limb motion vectors
        shoulder_vec = calculate_motion_vector(
            self.prev_norm_landmarks.shoulder_center,
            norm_landmarks.shoulder_center,
            dt,
        )
        hip_vec = calculate_motion_vector(
            self.prev_norm_landmarks.hip_center,
            norm_landmarks.hip_center,
            dt,
        )
        knee_vec = calculate_motion_vector(
            self.prev_norm_landmarks.knee_center,
            norm_landmarks.knee_center,
            dt,
        )
        ankle_vec = calculate_motion_vector(
            self.prev_norm_landmarks.ankle_center,
            norm_landmarks.ankle_center,
            dt,
        )
        center_vec = calculate_motion_vector(
            self.prev_center,
            curr_center,
            dt,
        )

        # 2. Append to rolling window
        if shoulder_vec.velocity is not None:
            self.shoulder_history.append(shoulder_vec.velocity)
        if hip_vec.velocity is not None:
            self.hip_history.append(hip_vec.velocity)
        if knee_vec.velocity is not None:
            self.knee_history.append(knee_vec.velocity)
        if ankle_vec.velocity is not None:
            self.ankle_history.append(ankle_vec.velocity)
        if center_vec.velocity is not None:
            self.center_history.append(center_vec.velocity)

        # 3. Compute smoothed velocities
        smooth_shoulder = self._mean(self.shoulder_history)
        smooth_hip = self._mean(self.hip_history)
        smooth_knee = self._mean(self.knee_history)
        smooth_ankle = self._mean(self.ankle_history)
        smooth_center = self._mean(self.center_history)

        limb_vals = [v for v in (smooth_shoulder, smooth_hip, smooth_knee, smooth_ankle) if v is not None]
        whole_body = float(np.mean(limb_vals)) if limb_vals else smooth_center

        profile = BodyMotionProfile(
            timestamp=timestamp,
            shoulder_velocity=smooth_shoulder,
            hip_velocity=smooth_hip,
            knee_velocity=smooth_knee,
            ankle_velocity=smooth_ankle,
            whole_body_motion=whole_body,
        )

        center_motion = BodyCenterMotion(
            timestamp=timestamp,
            x=float(curr_center[0]) if curr_center is not None else None,
            y=float(curr_center[1]) if curr_center is not None else None,
            displacement=center_vec.magnitude,
            velocity=smooth_center,
        )

        # Update previous frame cache
        self.prev_timestamp = timestamp
        self.prev_norm_landmarks = norm_landmarks
        self.prev_center = curr_center

        return profile, center_motion

    @staticmethod
    def _mean(history: Deque[float]) -> Optional[float]:
        if not history:
            return None
        return float(np.mean(list(history)))

    @staticmethod
    def _normalize_point(pt: Optional[np.ndarray], w: float, h: float) -> Optional[np.ndarray]:
        if pt is None or w <= 0 or h <= 0:
            return None
        return np.array([float(pt[0]) / w, float(pt[1]) / h], dtype=float)

    def _normalize_landmarks(self, lm: BodyLandmarks, w: float, h: float) -> BodyLandmarks:
        norm = lambda p: self._normalize_point(p, w, h)
        return BodyLandmarks(
            nose=norm(lm.nose),
            shoulder_center=norm(lm.shoulder_center),
            hip_center=norm(lm.hip_center),
            knee_center=norm(lm.knee_center),
            ankle_center=norm(lm.ankle_center),
            left_shoulder=norm(lm.left_shoulder),
            right_shoulder=norm(lm.right_shoulder),
            left_hip=norm(lm.left_hip),
            right_hip=norm(lm.right_hip),
            left_knee=norm(lm.left_knee),
            right_knee=norm(lm.right_knee),
            left_ankle=norm(lm.left_ankle),
            right_ankle=norm(lm.right_ankle),
        )
