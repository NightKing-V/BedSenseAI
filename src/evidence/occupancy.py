"""
Bed Occupancy & Spatial Affinity Feature Extraction for BedSense AI Evidence Engine.
Calculates multi-joint continuous spatial affinity against bed coordinates.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .geometry import BodyLandmarks


def normalize_point_in_bbox(
    point: Optional[np.ndarray],
    bbox: Optional[Tuple[float, float, float, float]],
) -> Optional[np.ndarray]:
    """
    Normalizes a 2D point into the bounding box coordinate frame [0, 1] x [0, 1].
    """
    if point is None or bbox is None:
        return None

    x, y = point[0], point[1]
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    if width <= 1e-6 or height <= 1e-6:
        return None

    norm_x = (x - x1) / width
    norm_y = (y - y1) / height

    return np.array([norm_x, norm_y], dtype=float)


def point_bed_boundary_distance(normalized_position: Optional[np.ndarray]) -> Optional[float]:
    """
    Calculates distance of normalized point to the bed bounding box boundary.
    Returns negative value if outside, positive distance if inside.
    """
    if normalized_position is None:
        return None

    x, y = float(normalized_position[0]), float(normalized_position[1])

    # Outside bed
    if x < 0:
        return float(x)
    if x > 1:
        return float(1.0 - x)
    if y < 0:
        return float(y)
    if y > 1:
        return float(1.0 - y)

    # Inside bed: distance to nearest edge
    distance = min(x, 1.0 - x, y, 1.0 - y)
    return float(distance)


def point_bed_affinity(normalized_position: Optional[np.ndarray]) -> Optional[float]:
    """
    Continuous affinity score [0.0, 1.0] for a point relative to bed.
    0.0 = completely outside or exactly at boundary.
    1.0 = deep inside bed center.
    """
    distance = point_bed_boundary_distance(normalized_position)
    if distance is None:
        return None

    if distance <= 0.0:
        return 0.0

    # Scales from 0 at the boundary up to 1.0 in the bed interior
    return float(min(1.0, distance * 2.0))


@dataclass
class BedRelativeFeatures:
    """Normalized landmark coordinates in bed space."""
    track_id: Optional[int] = None

    shoulder_position: Optional[np.ndarray] = None
    hip_position: Optional[np.ndarray] = None
    knee_position: Optional[np.ndarray] = None
    ankle_position: Optional[np.ndarray] = None


@dataclass
class BedAffinityFeatures:
    """Individual landmark affinities to bed."""
    track_id: Optional[int] = None

    shoulder_affinity: Optional[float] = None
    hip_affinity: Optional[float] = None
    knee_affinity: Optional[float] = None
    ankle_affinity: Optional[float] = None


@dataclass
class BedOccupancyEvidence:
    """Combined bed occupancy evidence scores."""
    shoulder: Optional[float] = None
    hip: Optional[float] = None
    knee: Optional[float] = None
    ankle: Optional[float] = None
    mean_affinity: Optional[float] = None


class BedOccupancyExtractor:
    """Extracts bed relative geometry and continuous multi-joint bed occupancy evidence."""

    def extract(
        self,
        landmarks: BodyLandmarks,
        bed_bbox: Optional[Tuple[float, float, float, float]],
        track_id: Optional[int] = None,
    ) -> Tuple[BedRelativeFeatures, BedOccupancyEvidence]:
        if bed_bbox is None:
            return (
                BedRelativeFeatures(track_id=track_id),
                BedOccupancyEvidence(mean_affinity=0.0),
            )

        rel_features = BedRelativeFeatures(
            track_id=track_id,
            shoulder_position=normalize_point_in_bbox(landmarks.shoulder_center, bed_bbox),
            hip_position=normalize_point_in_bbox(landmarks.hip_center, bed_bbox),
            knee_position=normalize_point_in_bbox(landmarks.knee_center, bed_bbox),
            ankle_position=normalize_point_in_bbox(landmarks.ankle_center, bed_bbox),
        )

        shoulder_aff = point_bed_affinity(rel_features.shoulder_position)
        hip_aff = point_bed_affinity(rel_features.hip_position)
        knee_aff = point_bed_affinity(rel_features.knee_position)
        ankle_aff = point_bed_affinity(rel_features.ankle_position)

        valid_affs = [v for v in (shoulder_aff, hip_aff, knee_aff, ankle_aff) if v is not None]
        mean_aff = float(np.mean(valid_affs)) if valid_affs else None

        evidence = BedOccupancyEvidence(
            shoulder=shoulder_aff,
            hip=hip_aff,
            knee=knee_aff,
            ankle=ankle_aff,
            mean_affinity=mean_aff,
        )

        return rel_features, evidence
