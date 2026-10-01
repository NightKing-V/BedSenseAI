"""
Data contracts for BedSense AI Monitoring System (§2 of Design Spec).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FrameObservation:
    """Primary Vision -> Temporal Engine contract (§2)."""
    t: float
    track_id: Optional[int]
    bbox: Optional[Tuple[float, float, float, float]]
    keypoints: Optional[np.ndarray]   # (17, 3) x, y, conf (COCO format)
    bed_overlap: float                # 0..1 overlap with bed area
    mean_kpt_conf: float              # average keypoint confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "track_id": self.track_id,
            "bbox": [round(v, 1) for v in self.bbox] if self.bbox else None,
            "bed_overlap": round(self.bed_overlap, 3),
            "mean_kpt_conf": round(self.mean_kpt_conf, 3),
        }


@dataclass
class StateSegment:
    """Temporal Engine -> Router / Agent LLM contract (§2)."""
    start_t: float
    end_t: float
    state: str
    confidence: float
    ambiguous_reason: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_t - self.start_t)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_t": round(self.start_t, 2),
            "end_t": round(self.end_t, 2),
            "duration": round(self.duration, 2),
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "ambiguous_reason": self.ambiguous_reason,
        }


@dataclass
class BedEvent:
    """Agent / Policy -> Output contract (§2 & §6.2)."""
    event: str                  # "bed_exit" | "bed_return"
    start_time: str             # "HH:MM:SS"
    confirmed_time: str         # "HH:MM:SS"
    previous_state: str         # lowercase e.g. "sitting_on_bed"
    current_state: str          # lowercase e.g. "walking"
    confidence: float           # 0..1
    decision: str               # "NORMAL" | "MONITOR" | "ALERT"
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_events_json_entry(self) -> Dict[str, Any]:
        """Strict schema matching §6.2."""
        return {
            "event": self.event,
            "start_time": self.start_time,
            "confirmed_time": self.confirmed_time,
            "previous_state": self.previous_state,
            "current_state": self.current_state,
            "confidence": round(self.confidence, 2),
            "decision": self.decision,
        }
