"""
State Machine & Pattern Matcher Subsystem for BedSense AI (§4.2 of Design Spec).
"""

from ..policy import PolicyEngine, ReportGenerator, format_hms
from ..temporal import FrameClassifier, TransitionSmoother, evaluate_ambiguity
from .matcher import BedPatternMatcher, CandidateEvent

__all__ = [
    "CandidateEvent",
    "BedPatternMatcher",
    # Backward compatibility re-exports
    "PolicyEngine",
    "ReportGenerator",
    "format_hms",
    "FrameClassifier",
    "TransitionSmoother",
    "evaluate_ambiguity",
]
