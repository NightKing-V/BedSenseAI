"""
BedSense AI — Vision -> Temporal Processing -> State Mechanism Monitoring System.
"""

from .constants import (
    ALL_STATES,
    DECISION_ALERT,
    DECISION_MONITOR,
    DECISION_NORMAL,
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    STATE_COLORS,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_UNKNOWN,
    STATE_WALKING,
)
from .contracts import BedEvent, FrameObservation, StateSegment
from .state import (
    BedPatternMatcher,
    CandidateEvent,
    PolicyEngine,
    ReportGenerator,
    format_hms,
)
from .temporal import (
    FrameClassifier,
    TransitionSmoother,
    evaluate_ambiguity,
    smooth_states,
)
from .vision import BedRelationEngine, PerceptionDetector

__all__ = [
    # Vision Layer
    "BedRelationEngine",
    "PerceptionDetector",
    # Temporal Processing Layer
    "FrameClassifier",
    "TransitionSmoother",
    "smooth_states",
    "evaluate_ambiguity",
    # State Mechanism & Policy Layer
    "BedPatternMatcher",
    "CandidateEvent",
    "PolicyEngine",
    "ReportGenerator",
    "format_hms",
    # Contracts
    "FrameObservation",
    "StateSegment",
    "BedEvent",
    # Constants
    "ALL_STATES",
    "STATE_LYING_IN_BED",
    "STATE_SITTING_ON_BED",
    "STATE_SITTING_OUTSIDE_BED",
    "STATE_STANDING",
    "STATE_WALKING",
    "STATE_OUT_OF_BED",
    "STATE_UNKNOWN",
    "EVENT_BED_EXIT",
    "EVENT_BED_RETURN",
    "DECISION_NORMAL",
    "DECISION_MONITOR",
    "DECISION_ALERT",
    "STATE_COLORS",
]
