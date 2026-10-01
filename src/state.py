"""
State Mechanism & Policy Engine for BedSense AI (§4, §6 of Design Spec).
Identifies bed-exit and bed-return sequences, evaluates alert policies, and generates JSON reports.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .constants import (
    DECISION_ALERT,
    DECISION_MONITOR,
    DECISION_NORMAL,
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_UNKNOWN,
    STATE_WALKING,
)
from .contracts import BedEvent, FrameObservation, StateSegment
from .temporal import FrameClassifier, TransitionSmoother, evaluate_ambiguity

logger = logging.getLogger("BedSenseState")


@dataclass
class CandidateEvent:
    event_type: str            # "bed_exit" | "bed_return"
    start_t: float
    confirmed_t: float
    prev_state: str
    curr_state: str
    confidence: float
    is_ambiguous: bool
    ambiguous_reason: Optional[str] = None


class BedPatternMatcher:
    """State machine identifying bed-exit and bed-return sequence patterns (§4.2)."""

    def __init__(self, min_exit_confirm_sec: float = 3.0, return_window_sec: float = 30.0):
        self.min_exit_confirm_sec = min_exit_confirm_sec
        self.return_window_sec = return_window_sec

    def find_candidates(self, segments: List[StateSegment]) -> List[CandidateEvent]:
        candidates: List[CandidateEvent] = []
        if len(segments) < 2:
            return candidates

        in_bed_states = {STATE_LYING_IN_BED, STATE_SITTING_ON_BED}
        out_states = {STATE_WALKING, STATE_OUT_OF_BED, STATE_SITTING_OUTSIDE_BED}
        in_bed = segments[0].state in in_bed_states

        i = 0
        while i < len(segments) - 1:
            prev_s = segments[i]
            next_s = segments[i + 1]

            # BED_EXIT: {LYING, SITTING} -> [STANDING] -> {WALKING, OUT_OF_BED}
            if in_bed and prev_s.state in in_bed_states:
                if next_s.state in out_states:
                    dur_out = next_s.duration
                    is_sustained = dur_out >= self.min_exit_confirm_sec
                    is_amb = not is_sustained or next_s.confidence < 0.65
                    candidates.append(CandidateEvent(
                        event_type=EVENT_BED_EXIT,
                        start_t=prev_s.end_t,
                        confirmed_t=next_s.start_t + min(dur_out, self.min_exit_confirm_sec),
                        prev_state=prev_s.state.lower(),
                        curr_state=next_s.state.lower(),
                        confidence=0.92 if is_sustained else 0.50,
                        is_ambiguous=is_amb,
                        ambiguous_reason="Bed exit duration short" if is_amb else None,
                    ))
                    in_bed = False
                elif next_s.state == STATE_STANDING and (i + 2 < len(segments)):
                    after_stand = segments[i + 2]
                    if after_stand.state in out_states:
                        dur_out = after_stand.duration
                        is_sustained = dur_out >= self.min_exit_confirm_sec
                        is_amb = not is_sustained or after_stand.confidence < 0.65
                        candidates.append(CandidateEvent(
                            event_type=EVENT_BED_EXIT,
                            start_t=prev_s.end_t,
                            confirmed_t=after_stand.start_t + min(dur_out, self.min_exit_confirm_sec),
                            prev_state=prev_s.state.lower(),
                            curr_state=after_stand.state.lower(),
                            confidence=0.94 if is_sustained else 0.55,
                            is_ambiguous=is_amb,
                            ambiguous_reason="Exit via standing" if is_amb else None,
                        ))
                        in_bed = False
                        i += 1
                    elif after_stand.state in in_bed_states:
                        # Rejection: Stood briefly and sat back down
                        i += 1

            # BED_RETURN: {WALKING, STANDING, OUT_OF_BED} -> {SITTING_ON_BED, LYING_IN_BED}
            elif not in_bed and (prev_s.state in out_states or prev_s.state == STATE_STANDING):
                if next_s.state in in_bed_states:
                    within_window = (next_s.start_t - prev_s.start_t) <= self.return_window_sec
                    is_amb = not within_window or next_s.confidence < 0.65
                    candidates.append(CandidateEvent(
                        event_type=EVENT_BED_RETURN,
                        start_t=prev_s.end_t,
                        confirmed_t=next_s.start_t + min(next_s.duration, 2.0),
                        prev_state=prev_s.state.lower(),
                        curr_state=next_s.state.lower(),
                        confidence=0.90 if within_window else 0.55,
                        is_ambiguous=is_amb,
                        ambiguous_reason="Bed return duration outside window" if is_amb else None,
                    ))
                    in_bed = True

            i += 1

        return candidates


class PolicyEngine:
    """
    Evaluates clinical alert policies and caregiver suppression (§6.1 of Design Spec).
    """

    def __init__(
        self,
        edge_sit_monitor_sec: float = 60.0,
        unknown_monitor_sec: float = 30.0,
        out_of_bed_alert_sec: float = 600.0,
        suppress_with_caregiver: bool = True,
    ):
        self.edge_sit_monitor_sec = edge_sit_monitor_sec
        self.unknown_monitor_sec = unknown_monitor_sec
        self.out_of_bed_alert_sec = out_of_bed_alert_sec
        self.suppress_with_caregiver = suppress_with_caregiver

    def evaluate_event(
        self,
        event_type: str,
        out_of_bed_sec: float = 0.0,
        has_caregiver: bool = False,
    ) -> str:
        """Determines decision tier: NORMAL | MONITOR | ALERT."""
        if event_type == EVENT_BED_RETURN:
            return DECISION_NORMAL

        decision = DECISION_MONITOR
        if out_of_bed_sec >= self.out_of_bed_alert_sec:
            decision = DECISION_ALERT

        if has_caregiver and self.suppress_with_caregiver:
            decision = self.suppress_escalation(decision)

        return decision

    @staticmethod
    def suppress_escalation(decision: str) -> str:
        """Suppresses one escalation tier if caregiver present."""
        if decision == DECISION_ALERT:
            return DECISION_MONITOR
        elif decision == DECISION_MONITOR:
            return DECISION_NORMAL
        return DECISION_NORMAL


def format_hms(seconds: Union[int, float]) -> str:
    """Format seconds into HH:MM:SS string (§6.2)."""
    sec = max(0, int(round(seconds)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class ReportGenerator:
    """
    Generates timeline.json, events.json, and summary.json strictly adhering to §6.2.
    Ensures sum(activity_duration_sec) == observation_duration_sec.
    """

    @staticmethod
    def build_summary(
        segments: List[StateSegment],
        events: List[BedEvent],
        total_obs_sec: float,
    ) -> Dict[str, Any]:
        obs_sec_int = max(0, int(round(total_obs_sec)))

        activity_map: Dict[str, int] = {
            "lying_in_bed": 0,
            "sitting_on_bed": 0,
            "sitting_outside_bed": 0,
            "standing": 0,
            "walking": 0,
            "unknown": 0,
        }

        in_bed_states = {STATE_LYING_IN_BED, STATE_SITTING_ON_BED}
        total_in_bed = 0.0
        total_out_bed = 0.0
        longest_out_period = 0.0
        current_out_period = 0.0

        for seg in segments:
            dur = seg.duration
            key = seg.state.lower()
            if key in activity_map:
                activity_map[key] += int(round(dur))
            elif key == "out_of_bed":
                activity_map["unknown"] += int(round(dur))

            if seg.state in in_bed_states:
                total_in_bed += dur
                if current_out_period > 0:
                    longest_out_period = max(longest_out_period, current_out_period)
                    current_out_period = 0.0
            else:
                total_out_bed += dur
                current_out_period += dur

        longest_out_period = max(longest_out_period, current_out_period)

        # Ensure exact integer sum conservation rule (§6.2)
        diff = obs_sec_int - sum(activity_map.values())
        if diff != 0:
            dominant_key = max(activity_map, key=activity_map.get)
            activity_map[dominant_key] = max(0, activity_map[dominant_key] + diff)

        bed_exits = sum(1 for e in events if e.event == EVENT_BED_EXIT)
        bed_returns = sum(1 for e in events if e.event == EVENT_BED_RETURN)
        final_state = segments[-1].state.lower() if segments else "unknown"

        return {
            "observation_duration_sec": obs_sec_int,
            "bed_exit_count": bed_exits,
            "bed_return_count": bed_returns,
            "total_in_bed_sec": int(round(total_in_bed)),
            "total_out_of_bed_sec": int(round(total_out_bed)),
            "longest_out_of_bed_period_sec": int(round(longest_out_period)),
            "final_state": final_state,
            "activity_duration_sec": activity_map,
        }

    @staticmethod
    def export(
        segments: List[StateSegment],
        events: List[BedEvent],
        output_dir: Union[str, Path],
        total_obs_sec: float,
    ):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timeline_data = [s.to_dict() for s in segments]
        with open(out_path / "timeline.json", "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)

        events_data = [e.to_events_json_entry() for e in events]
        with open(out_path / "events.json", "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2)

        summary_data = ReportGenerator.build_summary(segments, events, total_obs_sec)
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)


__all__ = [
    "CandidateEvent",
    "BedPatternMatcher",
    "PolicyEngine",
    "ReportGenerator",
    "format_hms",
    "FrameClassifier",
    "TransitionSmoother",
    "evaluate_ambiguity",
]
