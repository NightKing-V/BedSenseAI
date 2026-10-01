"""
Temporal Processing Layer for BedSense AI (§3 & §4 of Design Spec).
Handles posture kinematics classification, temporal transition smoothing, and ambiguity evaluation.
"""

from .ambiguity import evaluate_ambiguity
from .classifier import FrameClassifier
from .smoother import TransitionSmoother, smooth_states

__all__ = [
    "FrameClassifier",
    "TransitionSmoother",
    "smooth_states",
    "evaluate_ambiguity",
]
