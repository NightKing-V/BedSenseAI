"""
Temporal Ambiguity Scoring & Confidence Evaluation for BedSense AI (§4.1 of Design Spec).
"""

from typing import Any, List, Optional, Tuple
from ..constants import STATE_LYING_IN_BED, STATE_UNKNOWN
from ..contracts import FrameObservation, StateSegment


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


__all__ = ["evaluate_ambiguity"]
