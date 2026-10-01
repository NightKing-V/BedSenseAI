"""
Vision Subsystem for BedSense AI (§3.1, §3.2 of Design Spec).
Provides YOLO-based dynamic bed calibration and YOLO-Pose resident tracking.
"""

from .bed_engine import BedRelationEngine
from .detector import PerceptionDetector

__all__ = ["BedRelationEngine", "PerceptionDetector"]
