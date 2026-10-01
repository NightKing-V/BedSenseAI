"""
Temporal Processing Layer for BedSense AI (§3 & §4 of Design Spec).
Handles posture kinematics classification, temporal transition smoothing, and ambiguity evaluation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .constants import (
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
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
from .contracts import FrameObservation, StateSegment


class FrameClassifier:
    """
    Posture Geometry Frame Classifier (§3 & §8 of Design Spec).
    Maps posture geometry × bed overlap into the 7 canonical states:
    LYING_IN_BED | SITTING_ON_BED | SITTING_OUTSIDE_BED | STANDING | WALKING | OUT_OF_BED | UNKNOWN
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

    def reset(self):
        self.prev_center = None
        self.prev_t = None
        self.was_out_of_bed = False

    def classify(
        self,
        obs: FrameObservation,
        bed_engine: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """Classify raw observation into (canonical state, confidence) relative to bed plane."""
        if obs.track_id is None or obs.bbox is None or obs.keypoints is None:
            return (STATE_OUT_OF_BED if self.was_out_of_bed else STATE_UNKNOWN), 0.35

        if obs.mean_kpt_conf < self.min_kpt_conf_unknown:
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
                # Dot product >= 0.65 (angle <= 50 deg with bed longitudinal axis)
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

        # 6. Posture Determination
        overlap = obs.bed_overlap
        is_lying = False
        is_sitting = False
        is_upright = False

        if has_torso and torso_angle_vert > self.torso_angle_lying_thresh:
            is_lying = True
        elif is_aligned_with_bed_plane:
            is_lying = True
        elif aspect_ratio < self.aspect_ratio_lying_max and (not has_torso or torso_angle_vert > 40.0):
            is_lying = True
        elif aspect_ratio >= self.aspect_ratio_standing_min or (has_torso and torso_angle_vert <= self.torso_angle_upright_max):
            if mean_knee_angle is not None and mean_knee_angle < self.knee_angle_bent_max and aspect_ratio < 1.60:
                is_sitting = True
            else:
                is_upright = True
        else:
            is_sitting = True

        # 7. Map to 7 Canonical States using Bed Overlap
        if is_lying:
            state = STATE_LYING_IN_BED if overlap >= 0.35 else STATE_OUT_OF_BED
            conf = 0.92 if (overlap >= self.in_bed_thresh or overlap <= self.out_of_bed_thresh) else 0.68
        elif is_sitting:
            state = STATE_SITTING_ON_BED if overlap >= 0.35 else STATE_SITTING_OUTSIDE_BED
            conf = 0.90 if (overlap >= self.in_bed_thresh or overlap <= self.out_of_bed_thresh) else 0.65
        elif is_upright:
            state = STATE_WALKING if velocity > self.walking_velocity_thresh else STATE_STANDING
            conf = 0.90
        else:
            state = STATE_UNKNOWN
            conf = 0.40

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



class TransitionSmoother:
    """
    Stabilizes per-frame predictions into contiguous StateSegments.
    Collapses blips shorter than min_segment_sec (§3 of Design Spec).
    """

    def __init__(self, min_segment_sec: float = 1.0):
        self.min_segment_sec = min_segment_sec

    def smooth(self, raw_records: List[Tuple[float, str, float, FrameObservation]]) -> List[StateSegment]:
        if not raw_records:
            return []

        # Step 1: Initial grouping of consecutive identical states
        initial: List[StateSegment] = []
        curr_state = raw_records[0][1]
        start_t = raw_records[0][0]
        conf_sum = raw_records[0][2]
        count = 1

        for i in range(1, len(raw_records)):
            t, st, conf, _ = raw_records[i]
            if st == curr_state:
                conf_sum += conf
                count += 1
            else:
                initial.append(StateSegment(start_t=start_t, end_t=t, state=curr_state, confidence=conf_sum / count))
                curr_state = st
                start_t = t
                conf_sum = conf
                count = 1

        dt = (raw_records[-1][0] - raw_records[0][0]) / max(1, len(raw_records) - 1)
        initial.append(StateSegment(start_t=start_t, end_t=raw_records[-1][0] + dt, state=curr_state, confidence=conf_sum / count))

        if len(initial) <= 1:
            return initial

        # Step 2: Multi-pass smoothing to absorb blips below min_segment_sec
        smoothed = list(initial)
        for _ in range(5):
            new_smoothed: List[StateSegment] = []
            i = 0
            while i < len(smoothed):
                seg = smoothed[i]
                if seg.duration < self.min_segment_sec:
                    prev_s = new_smoothed[-1] if new_smoothed else None
                    next_s = smoothed[i + 1] if (i + 1 < len(smoothed)) else None

                    if prev_s and next_s and prev_s.state == next_s.state:
                        prev_s.end_t = seg.end_t
                        i += 1
                        continue
                    elif prev_s and (not next_s or (prev_s.duration >= next_s.duration)):
                        prev_s.end_t = seg.end_t
                        i += 1
                        continue
                    elif next_s:
                        next_s.start_t = seg.start_t
                        i += 1
                        continue

                new_smoothed.append(seg)
                i += 1

            # Merge adjacent identical states
            merged: List[StateSegment] = []
            for s in new_smoothed:
                if not merged:
                    merged.append(s)
                elif merged[-1].state == s.state:
                    merged[-1].end_t = s.end_t
                    merged[-1].confidence = (merged[-1].confidence + s.confidence) / 2.0
                else:
                    merged.append(s)
            smoothed = merged

        # Step 3: Ensure continuity
        for k in range(len(smoothed) - 1):
            smoothed[k].end_t = smoothed[k + 1].start_t

        return smoothed


def smooth_states(
    raw_records: List[Tuple[float, str, float, FrameObservation]],
    min_segment_sec: float = 2.0,
) -> List[StateSegment]:
    """Convenience function for temporal smoothing."""
    smoother = TransitionSmoother(min_segment_sec=min_segment_sec)
    return smoother.smooth(raw_records)


def evaluate_ambiguity(
    segment: StateSegment,
    observations: List[FrameObservation],
    candidate_events: Optional[List[Any]] = None,
) -> Tuple[bool, float, Optional[str]]:
    """
    Evaluates segment against §4.1 ambiguity triggers:
    1. mean_kpt_conf < 0.40
    2. Candidate bed-exit/return boundary uncertainty
    3. Bed overlap hovering in 0.40 - 0.60
    4. Conflict: horizontal posture detected off-bed (fall risk)
    5. Track ID loss and reacquisition
    """
    if not observations:
        return False, segment.confidence, None

    reasons: List[str] = []

    # 1. Low keypoint confidence
    kpt_confs = [o.mean_kpt_conf for o in observations if o.mean_kpt_conf > 0]
    avg_kpt = sum(kpt_confs) / max(1, len(kpt_confs)) if kpt_confs else 0.0
    if avg_kpt < 0.40 and segment.state != STATE_UNKNOWN:
        reasons.append(f"Low keypoint confidence (avg {avg_kpt:.2f} < 0.40)")

    # 2. Candidate event boundary
    if candidate_events:
        for ce in candidate_events:
            if (segment.start_t - 1.0 <= ce.start_t <= segment.end_t + 1.0) and getattr(ce, "is_ambiguous", False):
                reasons.append(f"Candidate {ce.event_type} boundary ambiguity: {getattr(ce, 'ambiguous_reason', 'uncertain')}")

    # 3. Bed boundary hover (0.40 - 0.60)
    overlaps = [o.bed_overlap for o in observations]
    hover_count = sum(1 for ov in overlaps if 0.40 <= ov <= 0.60)
    if hover_count / max(1, len(overlaps)) > 0.60 and segment.duration >= 3.0:
        reasons.append("Sustained hovering near bed boundary [0.4-0.6 overlap]")

    # 4. Horizontal off-bed (fall risk)
    avg_overlap = sum(overlaps) / max(1, len(overlaps))
    if segment.state == STATE_LYING_IN_BED and avg_overlap < 0.25:
        reasons.append("Conflict: horizontal lying posture detected outside bed region (potential fall)")

    # 5. Track ID lost and reacquired
    track_ids = [o.track_id for o in observations]
    if None in track_ids and track_ids[0] is not None and track_ids[-1] is not None:
        reasons.append("Track ID temporarily lost and re-acquired within segment")

    if reasons:
        return True, min(segment.confidence, 0.48), "; ".join(reasons)
    return False, segment.confidence, None
