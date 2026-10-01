"""
BedSense AI Evidence Engine Package.
Implements multi-attribute additive evidence accumulation, multi-joint motion sensing, and continuous bed occupancy affinity.
"""

from .geometry import (
    COCO_KEYPOINTS,
    BodyLandmarks,
    LandmarkExtractor,
    PoseFeatureExtractor,
    PoseFeatures,
    joint_angle,
    normalize_vector,
    point_distance,
    vector_between,
)
from .motion import (
    BodyCenterMotion,
    BodyMotionProfile,
    EvidenceMotionTracker,
    MotionVector,
    calculate_motion_vector,
    point_displacement,
)
from .occupancy import (
    BedAffinityFeatures,
    BedOccupancyEvidence,
    BedOccupancyExtractor,
    BedRelativeFeatures,
    normalize_point_in_bbox,
    point_bed_affinity,
    point_bed_boundary_distance,
)
from .classifier import (
    ActivityDecision,
    ActivityDecisionMaker,
    ActivityEvidence,
    EvidenceTemporalClassifier,
    RuleBasedEvidenceExtractor,
)

__all__ = [
    "COCO_KEYPOINTS",
    "BodyLandmarks",
    "LandmarkExtractor",
    "PoseFeatureExtractor",
    "PoseFeatures",
    "joint_angle",
    "normalize_vector",
    "point_distance",
    "vector_between",
    "BodyCenterMotion",
    "BodyMotionProfile",
    "EvidenceMotionTracker",
    "MotionVector",
    "calculate_motion_vector",
    "point_displacement",
    "BedAffinityFeatures",
    "BedOccupancyEvidence",
    "BedOccupancyExtractor",
    "BedRelativeFeatures",
    "normalize_point_in_bbox",
    "point_bed_affinity",
    "point_bed_boundary_distance",
    "ActivityDecision",
    "ActivityDecisionMaker",
    "ActivityEvidence",
    "EvidenceTemporalClassifier",
    "RuleBasedEvidenceExtractor",
]
