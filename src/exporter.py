"""
Decoupled Exporter & Video Annotation Module for BedSense AI.
Handles rendering of separate annotated videos for Method 1 (Heuristic) and Method 2 (Evidence),
JSON timeline/summary exports, CSV telemetry saving, and terminal comparison reports.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .constants import STATE_COLORS
from .contracts import BedEvent, FrameObservation, StateSegment
from .policy import ReportGenerator, format_hms

logger = logging.getLogger("BedSenseExporter")

# COCO 17 Skeleton Connection Pairs for visual skeleton overlay
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16),                  # Right leg
]


class VideoAnnotator:
    """Renders annotated visualization frames and encodes output MP4 videos."""

    @staticmethod
    def render_frame_heuristic(
        frame: np.ndarray,
        obs: FrameObservation,
        bed_engine: Any,
        live_m1: Dict[str, Any],
        metrics_m1: Dict[str, Any],
    ) -> np.ndarray:
        """Renders Method 1 (Heuristic Decision Tree) visual telemetry overlay."""
        vis = frame.copy()

        # 1. Draw Bed Region
        if bed_engine and bed_engine.polygon is not None:
            pts = bed_engine.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 220, 0), 2)
            cv2.putText(vis, "BED REGION", (pts[0][0][0], max(20, pts[0][0][1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

        # 2. Draw Person Bounding Box & Keypoints
        state_m1 = live_m1["live_state"]
        conf_m1 = live_m1["live_conf"]
        color = STATE_COLORS.get(state_m1, (255, 255, 255))

        if obs.bbox is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in obs.bbox]
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 2)
            label = f"M1: {state_m1} ({conf_m1*100:.0f}%)"
            cv2.putText(vis, label, (rx1, max(20, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if obs.keypoints is not None:
            for kx, ky, kc in obs.keypoints:
                if kc > 0.25:
                    cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)

        # 3. Method 1 HUD Banner
        hud_line = f"M1 [HEURISTIC] : {state_m1:<16} ({conf_m1*100:.0f}%) | BED OVERLAP: {obs.bed_overlap*100:4.1f}%"
        vel = metrics_m1.get("velocity", 0.0)
        hud_sub = f"2D Angle: {metrics_m1.get('torso_angle', 'N/A')} deg | Velocity: {vel:.1f} px/s"

        cv2.rectangle(vis, (10, 10), (580, 56), (20, 20, 20), -1)
        cv2.putText(vis, hud_line, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
        cv2.putText(vis, hud_sub, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        return vis

    @staticmethod
    def render_frame_evidence(
        frame: np.ndarray,
        obs: FrameObservation,
        bed_engine: Any,
        live_m2: Dict[str, Any],
        metrics_m2: Dict[str, Any],
    ) -> np.ndarray:
        """Renders Method 2 (Evidence-Based Biomechanics) visual telemetry overlay."""
        vis = frame.copy()

        # 1. Draw Bed Region
        if bed_engine and bed_engine.polygon is not None:
            pts = bed_engine.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 220, 0), 2)
            cv2.putText(vis, "BED REGION", (pts[0][0][0], max(20, pts[0][0][1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

        # 2. Draw Skeleton Bones & Keypoints
        if obs.keypoints is not None and len(obs.keypoints) == 17:
            kpts = obs.keypoints
            for pt1_idx, pt2_idx in SKELETON_EDGES:
                x1, y1, c1 = kpts[pt1_idx]
                x2, y2, c2 = kpts[pt2_idx]
                if c1 > 0.3 and c2 > 0.3:
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2)

            for kx, ky, kc in kpts:
                if kc > 0.3:
                    cv2.circle(vis, (int(kx), int(ky)), 4, (0, 255, 255), -1)

        # 3. Draw Person Bounding Box
        state_m2 = live_m2["live_state"]
        conf_m2 = live_m2["live_conf"]
        color = STATE_COLORS.get(state_m2, (255, 255, 255))

        if obs.bbox is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in obs.bbox]
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 2)
            label = f"M2: {state_m2} ({conf_m2*100:.0f}%)"
            cv2.putText(vis, label, (rx1, max(20, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 4. Method 2 Evidence HUD Overlay with Scores & Biomechanics
        aff = metrics_m2.get("mean_bed_affinity", 0.0)
        torso_n = metrics_m2.get("torso_length_normalized", "N/A")
        ev_ly = metrics_m2.get("evidence_lying", 0.0)
        ev_si = metrics_m2.get("evidence_sitting", 0.0)
        ev_st = metrics_m2.get("evidence_standing", 0.0)
        ev_wa = metrics_m2.get("evidence_walking", 0.0)

        hud_line1 = f"M2 [EVIDENCE]  : {state_m2:<16} ({conf_m2*100:.0f}%) | BED AFFINITY: {aff*100:4.1f}% | TorsoNorm: {torso_n}"
        hud_line2 = f"EVIDENCE: LYING={ev_ly*100:.0f}% | SITTING={ev_si*100:.0f}% | STANDING={ev_st*100:.0f}% | WALKING={ev_wa*100:.0f}%"

        cv2.rectangle(vis, (10, 10), (660, 60), (20, 20, 20), -1)
        cv2.putText(vis, hud_line1, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 1)
        cv2.putText(vis, hud_line2, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

        return vis

    @staticmethod
    def save_video(frames: List[np.ndarray], output_path: Union[str, Path], fps: float = 10.0):
        """Encodes a sequence of BGR image frames to an MP4 video file."""
        if not frames:
            return
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info(f"Annotated video saved successfully: {out_file.resolve()}")


class ArtifactExporter:
    """Manages all output exports: timelines, summaries, CSV telemetry, and annotated videos."""

    @staticmethod
    def export_primary(
        output_dir: Union[str, Path],
        segments: List[StateSegment],
        events: List[BedEvent],
        obs_duration: float,
        telemetry_records: List[Dict[str, Any]],
        frames: Optional[List[np.ndarray]] = None,
        video_name: str = "annotated.mp4",
        fps: float = 10.0,
    ):
        """Exports primary pipeline artifacts (Method 2 Evidence-Based Engine by default)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ReportGenerator.export(
            segments=segments,
            events=events,
            output_dir=out_path,
            total_obs_sec=obs_duration,
            telemetry_records=telemetry_records,
        )

        if frames:
            VideoAnnotator.save_video(frames, out_path / video_name, fps=fps)

        logger.info(
            f"Primary pipeline artifacts exported successfully to '{out_path}/':\n"
            f"  • timeline.json ({len(segments)} segments)\n"
            f"  • events.json ({len(events)} events)\n"
            f"  • summary.json\n"
            f"  • telemetry.csv ({len(telemetry_records)} frames)\n"
            f"  • {video_name}"
        )

    @staticmethod
    def export_all(
        output_dir: Union[str, Path],
        segments_m1: List[StateSegment],
        events_m1: List[BedEvent],
        segments_m2: List[StateSegment],
        events_m2: List[BedEvent],
        obs_duration: float,
        telemetry_records: List[Dict[str, Any]],
        frames_m1: Optional[List[np.ndarray]] = None,
        frames_m2: Optional[List[np.ndarray]] = None,
        fps: float = 10.0,
    ):
        """Exports side-by-side comparison artifacts for Method 1 and Method 2."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Primary timeline and summary default to Method 2 (Evidence)
        ReportGenerator.export(
            segments=segments_m2,
            events=events_m2,
            output_dir=out_path,
            total_obs_sec=obs_duration,
            telemetry_records=telemetry_records,
            segments_heuristic=segments_m1,
            events_heuristic=events_m1,
            segments_evidence=segments_m2,
            events_evidence=events_m2,
        )

        # Encode and save annotated videos
        if frames_m1:
            VideoAnnotator.save_video(frames_m1, out_path / "annotated_heuristic.mp4", fps=fps)
        if frames_m2:
            VideoAnnotator.save_video(frames_m2, out_path / "annotated_evidence.mp4", fps=fps)
            VideoAnnotator.save_video(frames_m2, out_path / "annotated.mp4", fps=fps)

        logger.info(
            f"Dual comparison artifacts exported successfully to '{out_path}/':\n"
            f"  • timeline.json (Primary: Method 2)\n"
            f"  • timeline_heuristic.json & timeline_evidence.json\n"
            f"  • summary.json & summary_comparison.json\n"
            f"  • telemetry.csv ({len(telemetry_records)} rows)\n"
            f"  • annotated_heuristic.mp4 & annotated_evidence.mp4"
        )

    @staticmethod
    def print_terminal_summary(
        obs_duration: float,
        segments: List[StateSegment],
        events: List[BedEvent],
        output_dir: Union[str, Path],
        method_name: str = "Method 2 (Evidence-Based Biomechanics)",
    ):
        """Prints a clean clinical monitoring summary for the active method."""
        summary = ReportGenerator.build_summary(segments, events, obs_duration)

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print("\n" + f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║         BEDSENSE AI CLINICAL STATE & MONITORING REPORT                               ║{RESET}")
        print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"Active Engine:             {GREEN}{method_name}{RESET}")
        print(f"Observation Duration:      {format_hms(obs_duration)} ({obs_duration:.1f}s)")
        print(f"Total Segments:            {len(segments)} | Final State: {summary['final_state'].upper()}")
        print(f"Bed Exits / Returns:       {summary['bed_exit_count']} exits / {summary['bed_return_count']} returns | Missing Alerts: {summary['missing_count']}")
        print(f"Time In Bed / Out Of Bed:  {format_hms(summary['total_in_bed_sec'])} ({summary['total_in_bed_sec']}s) / {format_hms(summary['total_out_of_bed_sec'])} ({summary['total_out_of_bed_sec']}s)")
        print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"{'Activity State':<28} | {'Duration (HH:MM:SS)':<22} | {'Percentage':<12}")
        print("-" * 70)
        for act, dur in summary["activity_duration_sec"].items():
            pct = (dur / max(1, obs_duration)) * 100
            print(f"{act.replace('_', ' ').title():<28} | {format_hms(dur)} ({dur:>3}s)          | {pct:5.1f}%")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}\n")
        print(f"Artifacts saved to: {Path(output_dir).resolve()}\n")

    @staticmethod
    def print_terminal_comparison(
        obs_duration: float,
        segments_m1: List[StateSegment],
        events_m1: List[BedEvent],
        segments_m2: List[StateSegment],
        events_m2: List[BedEvent],
        output_dir: Union[str, Path],
    ):
        """Prints a side-by-side terminal report comparing Method 1 and Method 2."""
        summary_m1 = ReportGenerator.build_summary(segments_m1, events_m1, obs_duration)
        summary_m2 = ReportGenerator.build_summary(segments_m2, events_m2, obs_duration)

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print("\n" + f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║         BEDSENSE AI DUAL TEMPORAL CLASSIFICATION COMPARISON REPORT                   ║{RESET}")
        print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"Observation Duration:      {format_hms(obs_duration)} ({obs_duration:.1f}s)")
        print(f"{YELLOW}Method 1 (Heuristic){RESET}  : {len(segments_m1)} segments | Final: {summary_m1['final_state'].upper()}")
        print(f"{GREEN}Method 2 (Evidence) {RESET}  : {len(segments_m2)} segments | Final: {summary_m2['final_state'].upper()}")
        print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"{'Activity State':<24} | {'M1 Heuristic Duration':<22} | {'M2 Evidence Duration':<22}")
        print("-" * 76)
        all_acts = sorted(list(set(summary_m1["activity_duration_sec"].keys()) | set(summary_m2["activity_duration_sec"].keys())))
        for act in all_acts:
            dur1 = summary_m1["activity_duration_sec"].get(act, 0)
            dur2 = summary_m2["activity_duration_sec"].get(act, 0)
            pct1 = (dur1 / max(1, obs_duration)) * 100
            pct2 = (dur2 / max(1, obs_duration)) * 100
            print(f"{act.replace('_', ' ').title():<24} | {format_hms(dur1)} ({dur1:>3}s | {pct1:4.1f}%)    | {format_hms(dur2)} ({dur2:>3}s | {pct2:4.1f}%)")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}\n")
        print(f"Artifacts and videos saved to: {Path(output_dir).resolve()}\n")
