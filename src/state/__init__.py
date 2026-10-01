"""
State Machine & Pattern Matcher Subsystem for BedSense AI (§4.2 of Design Spec).
"""

from ..policy import PolicyEngine, ReportGenerator, evaluate_ambiguity, format_hms
from .matcher import BedPatternMatcher, CandidateEvent

__all__ = [
    "CandidateEvent",
    "BedPatternMatcher",
    # Backward compatibility re-exports
    "PolicyEngine",
    "ReportGenerator",
    "format_hms",
    "evaluate_ambiguity",
]
