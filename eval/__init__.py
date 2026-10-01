"""
BedSense AI Evaluation Module.
Provides comprehensive benchmarking for Activity Recognition, Bed Events, and Duration Estimation.
"""

from .metrics import (
    MissingArtifactError,
    evaluate_single_sample,
    evaluate_multiple_samples,
    load_ground_truth,
    load_predictions,
)
from .report import (
    generate_markdown_report,
    print_terminal_evaluation,
)

__all__ = [
    "MissingArtifactError",
    "evaluate_single_sample",
    "evaluate_multiple_samples",
    "load_ground_truth",
    "load_predictions",
    "generate_markdown_report",
    "print_terminal_evaluation",
]
