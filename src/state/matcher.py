"""
State Machine Sequence Pattern Matcher for BedSense AI (§4.2 of Design Spec).
Identifies bed-exit and bed-return sequences across smoothed timeline segments.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..constants import (
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    EVENT_MISSING,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_UNKNOWN,
    STATE_WALKING,
)
from ..contracts import StateSegment


@dataclass
class CandidateEvent:
    """Represents a potential bed-exit, missing, or bed-return pattern transition before confirmation."""
    event_type: str            # "bed_exit" | "bed_return" | "missing"
    start_t: float
    confirmed_t: float
    prev_state: str
    curr_state: str
    confidence: float
    is_ambiguous: bool
    ambiguous_reason: Optional[str] = None


class BedPatternMatcher:
    """State machine identifying bed-exit, missing, and bed-return sequence patterns (§4.2)."""

    def __init__(
        self,
        min_exit_confirm_sec: float = 3.0,
        return_window_sec: float = 30.0,
        out_of_bed_alert_sec: float = 10.0,
    ):
        self.min_exit_confirm_sec = min_exit_confirm_sec
        self.return_window_sec = return_window_sec
        self.out_of_bed_alert_sec = out_of_bed_alert_sec

    def find_candidates(self, segments: List[StateSegment]) -> List[CandidateEvent]:
        """Scans continuous state segments for bed-exit, missing, and bed-return transitions."""
        candidates: List[CandidateEvent] = []
        if len(segments) < 2:
            return candidates

        in_bed_states = {STATE_LYING_IN_BED, STATE_SITTING_ON_BED}
        out_states = {STATE_WALKING, STATE_OUT_OF_BED, STATE_SITTING_OUTSIDE_BED, STATE_UNKNOWN}
        in_bed = segments[0].state in in_bed_states

        exit_start_t: Optional[float] = None
        missing_emitted = False

        i = 0
        while i < len(segments) - 1:
            prev_s = segments[i]
            next_s = segments[i + 1]

            # BED_EXIT: {LYING, SITTING} -> [STANDING] -> {WALKING, OUT_OF_BED, UNKNOWN}
            if in_bed and prev_s.state in in_bed_states:
                if next_s.state in out_states:
                    dur_out = next_s.duration
                    is_sustained = dur_out >= self.min_exit_confirm_sec
                    is_amb = not is_sustained or next_s.confidence < 0.65
                    exit_start_t = prev_s.end_t
                    missing_emitted = False
                    candidates.append(CandidateEvent(
                        event_type=EVENT_BED_EXIT,
                        start_t=prev_s.end_t,
                        confirmed_t=prev_s.end_t + self.min_exit_confirm_sec,
                        prev_state=prev_s.state.lower(),
                        curr_state=next_s.state.lower(),
                        confidence=next_s.confidence,
                        is_ambiguous=is_amb,
                        ambiguous_reason="Bed exit verification" if is_amb else None,
                    ))
                    in_bed = False

                    if next_s.state == STATE_UNKNOWN:
                        candidates.append(CandidateEvent(
                            event_type=EVENT_MISSING,
                            start_t=prev_s.end_t,
                            confirmed_t=prev_s.end_t + self.min_exit_confirm_sec,
                            prev_state=prev_s.state.lower(),
                            curr_state="unknown",
                            confidence=next_s.confidence,
                            is_ambiguous=False,
                        ))
                        missing_emitted = True
                    elif dur_out >= self.out_of_bed_alert_sec:
                        candidates.append(CandidateEvent(
                            event_type=EVENT_MISSING,
                            start_t=exit_start_t,
                            confirmed_t=exit_start_t + self.out_of_bed_alert_sec,
                            prev_state="out_of_bed",
                            curr_state=next_s.state.lower(),
                            confidence=next_s.confidence,
                            is_ambiguous=False,
                        ))
                        missing_emitted = True

                elif next_s.state == STATE_STANDING and (i + 2 < len(segments)):
                    after_stand = segments[i + 2]
                    if after_stand.state in out_states:
                        dur_out = after_stand.duration
                        is_sustained = dur_out >= self.min_exit_confirm_sec
                        is_amb = not is_sustained or after_stand.confidence < 0.65
                        exit_start_t = prev_s.end_t
                        missing_emitted = False
                        candidates.append(CandidateEvent(
                            event_type=EVENT_BED_EXIT,
                            start_t=prev_s.end_t,
                            confirmed_t=prev_s.end_t + self.min_exit_confirm_sec,
                            prev_state=prev_s.state.lower(),
                            curr_state=after_stand.state.lower(),
                            confidence=after_stand.confidence,
                            is_ambiguous=is_amb,
                            ambiguous_reason="Exit via standing" if is_amb else None,
                        ))
                        in_bed = False

                        if after_stand.state == STATE_UNKNOWN:
                            candidates.append(CandidateEvent(
                                event_type=EVENT_MISSING,
                                start_t=prev_s.end_t,
                                confirmed_t=prev_s.end_t + self.min_exit_confirm_sec,
                                prev_state=prev_s.state.lower(),
                                curr_state="unknown",
                                confidence=after_stand.confidence,
                                is_ambiguous=False,
                            ))
                            missing_emitted = True
                        elif (after_stand.end_t - exit_start_t) >= self.out_of_bed_alert_sec:
                            candidates.append(CandidateEvent(
                                event_type=EVENT_MISSING,
                                start_t=exit_start_t,
                                confirmed_t=exit_start_t + self.out_of_bed_alert_sec,
                                prev_state="out_of_bed",
                                curr_state=after_stand.state.lower(),
                                confidence=after_stand.confidence,
                                is_ambiguous=False,
                            ))
                            missing_emitted = True
                        i += 1
                    elif after_stand.state in in_bed_states:
                        # Rejection rule: Stood briefly and sat back down
                        i += 1

            # MISSING / BED_RETURN when out of bed
            elif not in_bed:
                # 1. Missing Rule: out_of_bed -> unknown (person not detected)
                if next_s.state == STATE_UNKNOWN and not missing_emitted:
                    candidates.append(CandidateEvent(
                        event_type=EVENT_MISSING,
                        start_t=prev_s.end_t,
                        confirmed_t=prev_s.end_t + self.min_exit_confirm_sec,
                        prev_state=prev_s.state.lower(),
                        curr_state="unknown",
                        confidence=next_s.confidence,
                        is_ambiguous=False,
                    ))
                    missing_emitted = True
                # 2. Missing Rule: prolonged out-of-bed absence >= out_of_bed_alert_sec
                elif exit_start_t is not None and (next_s.end_t - exit_start_t) >= self.out_of_bed_alert_sec and not missing_emitted:
                    candidates.append(CandidateEvent(
                        event_type=EVENT_MISSING,
                        start_t=exit_start_t,
                        confirmed_t=exit_start_t + self.out_of_bed_alert_sec,
                        prev_state="out_of_bed",
                        curr_state=next_s.state.lower(),
                        confidence=next_s.confidence,
                        is_ambiguous=False,
                    ))
                    missing_emitted = True

                # BED_RETURN: {WALKING, STANDING, OUT_OF_BED, UNKNOWN} -> {SITTING_ON_BED, LYING_IN_BED}
                if next_s.state in in_bed_states:
                    ref_t = exit_start_t if exit_start_t is not None else prev_s.start_t
                    within_window = (next_s.start_t - ref_t) <= self.return_window_sec
                    is_amb = not within_window or next_s.confidence < 0.65
                    candidates.append(CandidateEvent(
                        event_type=EVENT_BED_RETURN,
                        start_t=prev_s.end_t,
                        confirmed_t=prev_s.end_t + 2.0,
                        prev_state=prev_s.state.lower(),
                        curr_state=next_s.state.lower(),
                        confidence=next_s.confidence,
                        is_ambiguous=is_amb,
                        ambiguous_reason="Bed return duration outside window" if is_amb else None,
                    ))
                    in_bed = True
                    exit_start_t = None
                    missing_emitted = False

            i += 1

        return candidates


__all__ = ["CandidateEvent", "BedPatternMatcher"]
