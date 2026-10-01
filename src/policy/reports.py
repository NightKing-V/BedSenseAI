"""
Clinical Report Generation & JSON Exporters for BedSense AI (§6.2 of Design Spec).
Builds summary.json, timeline.json, and events.json adhering to clinical specifications.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..constants import (
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    EVENT_MISSING,
    STATE_LYING_IN_BED,
    STATE_SITTING_ON_BED,
)
from ..contracts import BedEvent, StateSegment

logger = logging.getLogger("BedSensePolicy")


def format_hms(seconds: Union[int, float]) -> str:
    """Formats seconds into standard HH:MM:SS string (§6.2)."""
    sec = max(0, int(round(seconds)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class ReportGenerator:
    """
    Generates timeline.json, events.json, summary.json, and telemetry.csv strictly adhering to §6.2.
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
        missing_count = sum(1 for e in events if e.event == EVENT_MISSING)
        final_state = segments[-1].state.lower() if segments else "unknown"

        return {
            "observation_duration_sec": obs_sec_int,
            "bed_exit_count": bed_exits,
            "bed_return_count": bed_returns,
            "missing_count": missing_count,
            "total_in_bed_sec": int(round(total_in_bed)),
            "total_out_of_bed_sec": int(round(total_out_bed)),
            "longest_out_of_bed_period_sec": int(round(longest_out_period)),
            "final_state": final_state,
            "activity_duration_sec": activity_map,
        }

    @staticmethod
    def export_csv(
        records: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        filename: str = "telemetry.csv",
    ) -> Path:
        """Exports frame telemetry records containing vision, posture, and temporal states to CSV."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_file = out_path / filename

        if not records:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                pass
            return csv_file

        # Preserve column sequence starting from the first record's keys
        fieldnames: List[str] = list(records[0].keys())
        for r in records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        return csv_file

    @staticmethod
    def export(
        segments: List[StateSegment],
        events: List[BedEvent],
        output_dir: Union[str, Path],
        total_obs_sec: float,
        telemetry_records: Optional[List[Dict[str, Any]]] = None,
        segments_heuristic: Optional[List[StateSegment]] = None,
        events_heuristic: Optional[List[BedEvent]] = None,
        segments_evidence: Optional[List[StateSegment]] = None,
        events_evidence: Optional[List[BedEvent]] = None,
    ):
        """
        Exports clinical timeline.json, events.json, summary.json, and telemetry.csv.
        If comparison segments are provided, also exports dual timelines and summary_comparison.json.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Primary timeline and summary (Method 2 by default)
        timeline_data = [s.to_dict() for s in segments]
        with open(out_path / "timeline.json", "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)

        events_data = [e.to_events_json_entry() for e in events]
        with open(out_path / "events.json", "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2)

        summary_data = ReportGenerator.build_summary(segments, events, total_obs_sec)
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # If running in dual comparison mode
        if segments_heuristic is not None and segments_evidence is not None:
            heur_timeline = [s.to_dict() for s in segments_heuristic]
            with open(out_path / "timeline_heuristic.json", "w", encoding="utf-8") as f:
                json.dump(heur_timeline, f, indent=2)

            evid_timeline = [s.to_dict() for s in segments_evidence]
            with open(out_path / "timeline_evidence.json", "w", encoding="utf-8") as f:
                json.dump(evid_timeline, f, indent=2)

            heur_events = events_heuristic or []
            with open(out_path / "events_heuristic.json", "w", encoding="utf-8") as f:
                json.dump([e.to_events_json_entry() for e in heur_events], f, indent=2)

            evid_events = events_evidence or []
            with open(out_path / "events_evidence.json", "w", encoding="utf-8") as f:
                json.dump([e.to_events_json_entry() for e in evid_events], f, indent=2)

            heur_summary = ReportGenerator.build_summary(segments_heuristic, heur_events, total_obs_sec)
            with open(out_path / "summary_heuristic.json", "w", encoding="utf-8") as f:
                json.dump(heur_summary, f, indent=2)

            evid_summary = ReportGenerator.build_summary(segments_evidence, evid_events, total_obs_sec)
            with open(out_path / "summary_evidence.json", "w", encoding="utf-8") as f:
                json.dump(evid_summary, f, indent=2)

            comparison_summary = {
                "observation_duration_sec": int(round(total_obs_sec)),
                "method_1_heuristic": heur_summary,
                "method_2_evidence": evid_summary,
            }
            with open(out_path / "summary_comparison.json", "w", encoding="utf-8") as f:
                json.dump(comparison_summary, f, indent=2)

        if telemetry_records is not None:
            ReportGenerator.export_csv(telemetry_records, output_dir, filename="telemetry.csv")


__all__ = ["ReportGenerator", "format_hms"]
