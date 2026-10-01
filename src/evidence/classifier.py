"""
Evidence-Based Activity Classifier for BedSense AI.
Implements multi-attribute additive evidence accumulation (pose geometry, limb motion, continuous bed affinity).
Directly ports and extends the evidence-based decision framework from notebookv2.ipynb.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..constants import (
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_UNKNOWN,
    STATE_WALKING,
)
from ..contracts import FrameObservation
from .geometry import BodyLandmarks, LandmarkExtractor, PoseFeatureExtractor, PoseFeatures
from .motion import BodyCenterMotion, BodyMotionProfile, EvidenceMotionTracker
from .occupancy import BedOccupancyEvidence, BedOccupancyExtractor, BedRelativeFeatures


@dataclass
class ActivityEvidence:
    """Normalized evidence scores [0.0, 1.0] across canonical postural states."""
    lying: float = 0.0
    sitting: float = 0.0
    standing: float = 0.0
    walking: float = 0.0
    unknown: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "lying": round(self.lying, 3),
            "sitting": round(self.sitting, 3),
            "standing": round(self.standing, 3),
            "walking": round(self.walking, 3),
            "unknown": round(self.unknown, 3),
        }


@dataclass
class ActivityDecision:
    """Final activity decision with winner state and confidence."""
    raw_state: str
    canonical_state: str
    confidence: float
    evidence: ActivityEvidence


class RuleBasedEvidenceExtractor:
    """
    Additive Evidence Accumulator for Human Activity Recognition.
    Computes soft probabilistic weights based on motion, joint geometry, and bed affinity.
    """

    def __init__(
        self,
        low_body_motion: float = 0.015,
        high_body_motion: float = 0.035,
        extended_knee_angle: float = 160.0,
        bent_knee_angle: float = 130.0,
        extended_hip_angle: float = 160.0,
        bent_hip_angle: float = 130.0,
        on_bed_threshold: float = 0.25,
        off_bed_threshold: float = 0.12,
        torso_length_normalized_threshold: float = 1.25,
    ):
        self.low_body_motion = low_body_motion
        self.high_body_motion = high_body_motion

        self.extended_knee_angle = extended_knee_angle
        self.bent_knee_angle = bent_knee_angle

        self.extended_hip_angle = extended_hip_angle
        self.bent_hip_angle = bent_hip_angle

        self.on_bed_threshold = on_bed_threshold
        self.off_bed_threshold = off_bed_threshold
        self.torso_length_normalized_threshold = torso_length_normalized_threshold

    def extract(
        self,
        pose_features: PoseFeatures,
        motion_profile: BodyMotionProfile,
        center_motion: BodyCenterMotion,
        occupancy: BedOccupancyEvidence,
        bed_overlap: float = 0.0,
    ) -> ActivityEvidence:
        # 1. Basic measurements
        body_vel = center_motion.velocity
        ankle_vel = motion_profile.ankle_velocity
        knee_vel = motion_profile.knee_velocity

        knee_angle = pose_features.mean_knee_angle
        hip_angle = pose_features.mean_hip_angle

        # 2. Determine bed relationship (continuous affinity + bbox overlap)
        mean_affinity = occupancy.mean_affinity
        if mean_affinity is not None:
            on_bed = (mean_affinity >= self.on_bed_threshold) or (bed_overlap >= 0.30)
            off_bed = (mean_affinity <= self.off_bed_threshold) and (bed_overlap < 0.20)
        else:
            on_bed = bed_overlap >= 0.35
            off_bed = bed_overlap <= 0.15

        # Determine posture geometry
        torso_inc = pose_features.torso_inclination_angle
        is_horizontal = (torso_inc is not None and torso_inc <= 45.0)

        has_bent_hip = (hip_angle is not None and hip_angle < self.bent_hip_angle)
        has_extended_hip = (hip_angle is not None and hip_angle >= self.extended_hip_angle)

        has_bent_knee = (knee_angle is not None and knee_angle < self.bent_knee_angle)
        has_extended_knee = (knee_angle is not None and knee_angle >= self.extended_knee_angle)

        # 3. Missing geometry fallback
        if knee_angle is None and hip_angle is None and body_vel is None and mean_affinity is None:
            return ActivityEvidence(unknown=1.0)

        lying = 0.0
        sitting = 0.0
        standing = 0.0
        walking = 0.0

        # --------------------------------------------------
        # 4. LYING EVIDENCE (Dominates when on_bed and horizontal/resting)
        # --------------------------------------------------
        if on_bed:
            if is_horizontal:
                # Transverse/lateral in bed
                lying += 0.70
                if body_vel is not None and body_vel < self.low_body_motion:
                    lying += 0.20
                if has_extended_knee or has_extended_hip:
                    lying += 0.10
            elif has_bent_hip:
                # Sitting upright in bed with flexed hips -> very low lying score
                lying += 0.05
            else:
                # Longitudinal/angled in bed: extended body or resting posture
                lying += 0.60
                if has_extended_hip:
                    lying += 0.20
                if has_extended_knee:
                    lying += 0.10
                if body_vel is not None and body_vel < self.low_body_motion:
                    lying += 0.15
        elif is_horizontal:
            # Horizontal outside bed (floor/fall)
            lying += 0.40
            if body_vel is not None and body_vel < self.low_body_motion:
                lying += 0.30

        # --------------------------------------------------
        # 5. SITTING EVIDENCE (Sitting on Bed or Chair)
        # --------------------------------------------------
        if on_bed:
            if has_bent_hip:
                # Clear biomechanical sitting: hip flexed < 130 deg
                sitting += 0.70
                if has_bent_knee:
                    sitting += 0.20
                if body_vel is not None and body_vel < self.low_body_motion:
                    sitting += 0.10
            elif has_bent_knee and not has_extended_hip:
                # Flexed knees with non-extended hips
                sitting += 0.45
                if body_vel is not None and body_vel < self.low_body_motion:
                    sitting += 0.15
            else:
                # Extended hips/legs or resting flat in bed
                sitting += 0.05
        elif not off_bed:
            # Near bed boundary / transition zone
            if has_bent_hip or has_bent_knee:
                sitting += 0.55
                if body_vel is not None and body_vel < self.low_body_motion:
                    sitting += 0.20
            elif not is_horizontal and (body_vel is not None and body_vel < self.low_body_motion):
                sitting += 0.20
        else:
            # Clear off bed (e.g. sitting on chair)
            if has_bent_hip or has_bent_knee:
                sitting += 0.60
                if body_vel is not None and body_vel < self.low_body_motion:
                    sitting += 0.25

        # --------------------------------------------------
        # 6. STANDING EVIDENCE (Requires clearing the bed + upright stationary posture)
        # --------------------------------------------------
        if not on_bed and not is_horizontal:
            if body_vel is not None and body_vel < self.low_body_motion:
                if off_bed:
                    standing += 0.40
                else:
                    standing += 0.20
                if has_extended_hip:
                    standing += 0.25
                if has_extended_knee:
                    standing += 0.25
                if not has_bent_hip and not has_bent_knee:
                    standing += 0.10

        # --------------------------------------------------
        # 7. WALKING EVIDENCE (Requires clearing the bed + upright locomotion motion)
        # --------------------------------------------------
        if not on_bed and not is_horizontal:
            if off_bed:
                walking += 0.30
            else:
                walking += 0.10

            if body_vel is not None and body_vel > self.high_body_motion:
                walking += 0.40
            if ankle_vel is not None and ankle_vel > self.high_body_motion:
                walking += 0.25
            if knee_vel is not None and knee_vel > self.high_body_motion:
                walking += 0.20

        # --------------------------------------------------
        # 8. Normalize Evidence Scores
        # --------------------------------------------------
        scores = [lying, sitting, standing, walking]
        total = sum(scores)

        if total <= 0:
            return ActivityEvidence(unknown=1.0)

        return ActivityEvidence(
            lying=lying / total,
            sitting=sitting / total,
            standing=standing / total,
            walking=walking / total,
            unknown=0.0,
        )


class ActivityDecisionMaker:
    """
    Decides the final state from normalized evidence scores and maps to canonical clinical states.
    """

    def __init__(self, minimum_confidence: float = 0.30):
        self.minimum_confidence = minimum_confidence

    def decide(
        self,
        evidence: ActivityEvidence,
        on_bed: bool = False,
        bed_overlap: float = 0.0,
    ) -> ActivityDecision:
        scores = {
            "LYING": evidence.lying,
            "SITTING": evidence.sitting,
            "STANDING": evidence.standing,
            "WALKING": evidence.walking,
        }

        best_raw_state = max(scores, key=scores.get)
        confidence = scores[best_raw_state]

        if confidence < self.minimum_confidence:
            return ActivityDecision(
                raw_state="UNKNOWN",
                canonical_state=STATE_UNKNOWN,
                confidence=confidence,
                evidence=evidence,
            )

        # Canonical mapping based on spatial bed relationship
        is_in_bed_zone = on_bed or bed_overlap >= 0.25

        if best_raw_state == "LYING":
            canonical = STATE_LYING_IN_BED
        elif best_raw_state == "SITTING":
            canonical = STATE_SITTING_ON_BED if is_in_bed_zone else STATE_SITTING_OUTSIDE_BED
        elif best_raw_state == "STANDING":
            canonical = STATE_STANDING
        elif best_raw_state == "WALKING":
            canonical = STATE_WALKING
        else:
            canonical = STATE_UNKNOWN

        return ActivityDecision(
            raw_state=best_raw_state,
            canonical_state=canonical,
            confidence=confidence,
            evidence=evidence,
        )


class EvidenceTemporalClassifier:
    """
    Unified Evidence-Based Temporal Classifier.
    Extracts geometric landmarks, continuous bed occupancy affinities, and multi-joint motion kinematics.
    """

    def __init__(
        self,
        min_keypoint_conf: float = 0.25,
        window_size: int = 5,
        minimum_confidence: float = 0.30,
        evidence_extractor: Optional[RuleBasedEvidenceExtractor] = None,
    ):
        self.landmark_extractor = LandmarkExtractor(min_confidence=min_keypoint_conf)
        self.pose_extractor = PoseFeatureExtractor()
        self.occupancy_extractor = BedOccupancyExtractor()
        self.motion_tracker = EvidenceMotionTracker(window_size=window_size)
        self.evidence_extractor = evidence_extractor or RuleBasedEvidenceExtractor()
        self.decision_maker = ActivityDecisionMaker(minimum_confidence=minimum_confidence)

        self.last_decision: Optional[ActivityDecision] = None
        self.last_metrics: Dict[str, Any] = {}

    def reset(self):
        self.motion_tracker.reset()
        self.last_decision = None
        self.last_metrics.clear()

    def classify(
        self,
        observation: FrameObservation,
        bed_engine: Any = None,
        frame_width: float = 1280.0,
        frame_height: float = 720.0,
    ) -> Tuple[str, float]:
        """
        Classifies the frame observation using the evidence-based accumulation framework.
        Returns:
            (canonical_state, confidence)
        """
        # 1. No person detected
        if observation.track_id is None or observation.bbox is None:
            self.last_decision = ActivityDecision(
                raw_state="UNKNOWN",
                canonical_state=STATE_UNKNOWN,
                confidence=0.0,
                evidence=ActivityEvidence(unknown=1.0),
            )
            self.last_metrics = {
                "evidence_lying": 0.0,
                "evidence_sitting": 0.0,
                "evidence_standing": 0.0,
                "evidence_walking": 0.0,
                "mean_bed_affinity": 0.0,
                "torso_length_normalized": 0.0,
                "body_center_velocity": 0.0,
                "ankle_velocity": 0.0,
                "knee_velocity": 0.0,
                "decision_raw": "UNKNOWN",
            }
            return STATE_UNKNOWN, 0.0

        # 2. Extract Geometry & Landmarks
        landmarks = self.landmark_extractor.extract(observation.keypoints)
        pose_features = self.pose_extractor.extract(landmarks, track_id=observation.track_id)

        # 3. Bed Occupancy & Relative Affinities
        bed_bbox = bed_engine.bbox if (bed_engine and bed_engine.bbox) else None
        rel_features, occupancy = self.occupancy_extractor.extract(
            landmarks,
            bed_bbox=bed_bbox,
            track_id=observation.track_id,
        )

        # 4. Multi-Joint Motion Sensing & Kinematics
        motion_profile, center_motion = self.motion_tracker.update(
            timestamp=observation.t,
            landmarks=landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # 5. Extract Evidence Scores
        evidence = self.evidence_extractor.extract(
            pose_features=pose_features,
            motion_profile=motion_profile,
            center_motion=center_motion,
            occupancy=occupancy,
            bed_overlap=observation.bed_overlap,
        )

        # 6. Make Canonical Activity Decision
        on_bed = (
            (occupancy.mean_affinity is not None and occupancy.mean_affinity >= self.evidence_extractor.on_bed_threshold)
            or observation.bed_overlap >= 0.30
        )
        decision = self.decision_maker.decide(
            evidence=evidence,
            on_bed=on_bed,
            bed_overlap=observation.bed_overlap,
        )

        self.last_decision = decision
        self.last_metrics = {
            "evidence_lying": round(evidence.lying, 3),
            "evidence_sitting": round(evidence.sitting, 3),
            "evidence_standing": round(evidence.standing, 3),
            "evidence_walking": round(evidence.walking, 3),
            "evidence_unknown": round(evidence.unknown, 3),
            "mean_bed_affinity": round(occupancy.mean_affinity, 3) if occupancy.mean_affinity is not None else 0.0,
            "shoulder_affinity": round(occupancy.shoulder, 3) if occupancy.shoulder is not None else "",
            "hip_affinity": round(occupancy.hip, 3) if occupancy.hip is not None else "",
            "knee_affinity": round(occupancy.knee, 3) if occupancy.knee is not None else "",
            "ankle_affinity": round(occupancy.ankle, 3) if occupancy.ankle is not None else "",
            "torso_length_normalized": round(pose_features.torso_length_normalized, 3) if pose_features.torso_length_normalized is not None else "",
            "knee_angle_deg": round(pose_features.mean_knee_angle, 1) if pose_features.mean_knee_angle is not None else "",
            "hip_angle_deg": round(pose_features.mean_hip_angle, 1) if pose_features.mean_hip_angle is not None else "",
            "body_center_velocity": round(center_motion.velocity, 4) if center_motion.velocity is not None else 0.0,
            "ankle_velocity": round(motion_profile.ankle_velocity, 4) if motion_profile.ankle_velocity is not None else 0.0,
            "knee_velocity": round(motion_profile.knee_velocity, 4) if motion_profile.knee_velocity is not None else 0.0,
            "decision_raw": decision.raw_state,
        }

        return decision.canonical_state, decision.confidence

    def get_last_metrics(self) -> Dict[str, Any]:
        return self.last_metrics
