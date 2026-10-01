"""
Transition Smoother for BedSense AI (§3 of Design Spec).
Stabilizes per-frame predictions into contiguous StateSegments by collapsing short transient blips.
"""

from typing import List, Tuple
from ..contracts import FrameObservation, StateSegment


class TransitionSmoother:
    """
    Stabilizes per-frame predictions into contiguous StateSegments.
    Collapses blips shorter than min_segment_sec (§3 of Design Spec).
    """

    def __init__(self, min_segment_sec: float = 1.0):
        self.min_segment_sec = min_segment_sec

    def smooth(self, raw_records: List[Tuple[float, str, float, FrameObservation]]) -> List[StateSegment]:
        if not raw_records:
            return []

        # Step 1: Initial grouping of consecutive identical states
        initial: List[StateSegment] = []
        curr_state = raw_records[0][1]
        start_t = raw_records[0][0]
        conf_sum = raw_records[0][2]
        count = 1

        for i in range(1, len(raw_records)):
            t, st, conf, _ = raw_records[i]
            if st == curr_state:
                conf_sum += conf
                count += 1
            else:
                initial.append(StateSegment(start_t=start_t, end_t=t, state=curr_state, confidence=conf_sum / count))
                curr_state = st
                start_t = t
                conf_sum = conf
                count = 1

        dt = (raw_records[-1][0] - raw_records[0][0]) / max(1, len(raw_records) - 1)
        initial.append(StateSegment(start_t=start_t, end_t=raw_records[-1][0] + dt, state=curr_state, confidence=conf_sum / count))

        if len(initial) <= 1:
            return initial

        # Step 2: Multi-pass smoothing to absorb blips below min_segment_sec
        smoothed = list(initial)
        for _ in range(5):
            new_smoothed: List[StateSegment] = []
            i = 0
            while i < len(smoothed):
                seg = smoothed[i]
                if seg.duration < self.min_segment_sec:
                    prev_s = new_smoothed[-1] if new_smoothed else None
                    next_s = smoothed[i + 1] if (i + 1 < len(smoothed)) else None

                    if prev_s and next_s and prev_s.state == next_s.state:
                        prev_s.end_t = seg.end_t
                        i += 1
                        continue
                    elif prev_s and (not next_s or (prev_s.duration >= next_s.duration)):
                        prev_s.end_t = seg.end_t
                        i += 1
                        continue
                    elif next_s:
                        next_s.start_t = seg.start_t
                        i += 1
                        continue

                new_smoothed.append(seg)
                i += 1

            # Merge adjacent identical states
            merged: List[StateSegment] = []
            for s in new_smoothed:
                if not merged:
                    merged.append(s)
                elif merged[-1].state == s.state:
                    merged[-1].end_t = s.end_t
                    merged[-1].confidence = (merged[-1].confidence + s.confidence) / 2.0
                else:
                    merged.append(s)
            smoothed = merged

        # Step 3: Ensure continuity
        for k in range(len(smoothed) - 1):
            smoothed[k].end_t = smoothed[k + 1].start_t

        return smoothed


def smooth_states(
    raw_records: List[Tuple[float, str, float, FrameObservation]],
    min_segment_sec: float = 2.0,
) -> List[StateSegment]:
    """Convenience function for temporal smoothing."""
    smoother = TransitionSmoother(min_segment_sec=min_segment_sec)
    return smoother.smooth(raw_records)


__all__ = ["TransitionSmoother", "smooth_states"]
