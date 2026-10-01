"""
Command-Line Interface for BedSense AI Monitoring System.
Executes the unified pipeline: Vision -> Temporal Processing -> State Mechanism.

Provides multi-level logging with high-visibility state transitions & bed events.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from .constants import (
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    STATE_COLORS,
    STATE_UNKNOWN,
)
from .contracts import BedEvent, FrameObservation, StateSegment
from .state import (
    BedPatternMatcher,
    PolicyEngine,
    ReportGenerator,
    format_hms,
)
from .temporal import FrameClassifier, TransitionSmoother
from .vision import BedRelationEngine, PerceptionDetector

# ---------------------------------------------------------------------------
# Multi-Level Console Logging Formatter
# ---------------------------------------------------------------------------
STATE_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(STATE_LEVEL_NUM, "STATE")


def log_state(self, message, *args, **kws):
    if self.isEnabledFor(STATE_LEVEL_NUM):
        self._log(STATE_LEVEL_NUM, message, args, **kws)


logging.Logger.state = log_state


class BedSenseFormatter(logging.Formatter):
    """Custom formatter distinguishing State Transitions, Operational Info, and Debug logs."""

    # ANSI colors for terminal
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == STATE_LEVEL_NUM:
            # High-visibility state transition / event formatting
            return f"{self.BOLD}{self.GREEN}[STATE/EVENT]{self.RESET} {record.getMessage()}"
        elif record.levelno >= logging.WARNING:
            return f"{self.BOLD}{self.YELLOW}[WARNING]{self.RESET} {record.getMessage()}"
        elif record.levelno == logging.INFO:
            return f"{self.CYAN}[INFO]{self.RESET} {record.getMessage()}"
        elif record.levelno <= logging.DEBUG:
            return f"{self.DIM}[DEBUG]{self.RESET} {record.getMessage()}"
        return super().format(record)


logger = logging.getLogger("BedSense")


def setup_logger(log_level_str: str):
    logger.handlers.clear()
    logger.propagate = False

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "STATE_ONLY": STATE_LEVEL_NUM,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level = level_map.get(log_level_str.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(BedSenseFormatter())
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Video Processing Pipeline
# ---------------------------------------------------------------------------
def process_video_cli(
    video_path: str,
    output_dir: str = "outputs",
    annotated_video: Optional[str] = None,
    fps: float = 10.0,
    max_frames: Optional[int] = None,
    min_segment_sec: float = 2.0,
    device: Optional[str] = None,
    log_level: str = "INFO",
):
    setup_logger(log_level)

    # Select compute device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using compute device: {device}")
    logger.info("Initializing Vision Layer (Dynamic Bed Calibration + YOLO-Pose Tracking)...")

    bed_engine = BedRelationEngine(
        model_path="yolo11n.pt",
        confidence_thresh=0.12,
        device=device,
    )
    detector = PerceptionDetector(
        model_path="yolo11n-pose.pt",
        conf_threshold=0.25,
        device=device,
        bed_engine=bed_engine,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    # 1. Vision: Auto-calibrate Bed Quad
    logger.info("Running dynamic bed detection across sample frames...")
    detector.auto_calibrate(video_path, sample_frames=15)
    if bed_engine.bbox is not None:
        bx1, by1, bx2, by2 = [round(v, 1) for v in bed_engine.bbox]
        logger.info(f"Calibrated Bed Bounding Box: ({bx1}, {by1}) -> ({bx2}, {by2})")
    else:
        logger.warning("Bed detection fallback: default proportional region.")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur_sec = total_frames / native_fps if total_frames > 0 else 0.0
    frame_interval = max(1, int(round(native_fps / fps)))

    logger.info(f"Video input: {video_path} ({total_frames} frames, {format_hms(total_dur_sec)} duration @ {native_fps:.1f} FPS)")
    logger.info(f"Processing rate: {fps:.1f} FPS (every {frame_interval} frames)")

    # 2. Pipeline Engines
    classifier = FrameClassifier()
    smoother = TransitionSmoother(min_segment_sec=min_segment_sec)
    pattern_matcher = BedPatternMatcher(min_exit_confirm_sec=3.0, return_window_sec=30.0)
    policy = PolicyEngine(
        edge_sit_monitor_sec=60.0,
        unknown_monitor_sec=30.0,
        out_of_bed_alert_sec=600.0,
        suppress_with_caregiver=True,
    )

    observations: List[FrameObservation] = []
    raw_records: List[Tuple[float, str, float, FrameObservation]] = []
    annotated_frames: List[np.ndarray] = []

    last_reported_state: Optional[str] = None
    state_start_ts: float = 0.0

    frame_idx = 0
    sampled_count = 0
    start_wall_time = time.time()

    logger.info("Starting Video Inference Pipeline: Vision -> Temporal Processing -> State Mechanism...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts = round(frame_idx / native_fps, 2)

            # Vision Detection & Tracking
            obs = detector.get_primary_observation(frame, ts)

            # Temporal Posture Classification (relative to bed plane)
            state, conf = classifier.classify(obs, bed_engine=bed_engine)

            observations.append(obs)
            raw_records.append((ts, state, conf, obs))

            # Debug Telemetry Log
            logger.debug(
                f"[t={ts:06.2f}s] State: {state:<18} | Conf: {conf*100:4.1f}% | "
                f"Track: {str(obs.track_id):<4} | Overlap: {obs.bed_overlap*100:4.1f}% | "
                f"KptConf: {obs.mean_kpt_conf:4.2f}"
            )

            # Initial state announcement
            if last_reported_state is None:
                logger.state(f"Initial State Identified at {format_hms(ts)}: {state} (Confidence: {conf*100:.0f}%)")
                last_reported_state = state
                state_start_ts = ts


            # Annotated Visual Frame Generation
            if annotated_video:
                vis_frame = frame.copy()
                if bed_engine.polygon is not None:
                    pts = bed_engine.polygon.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_frame, [pts], True, (0, 200, 0), 2)

                if obs.bbox is not None:
                    rx1, ry1, rx2, ry2 = [int(v) for v in obs.bbox]
                    color = STATE_COLORS.get(state, (255, 255, 255))
                    cv2.rectangle(vis_frame, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(vis_frame, f"{state} (ID:{obs.track_id})", (rx1, max(18, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if obs.keypoints is not None:
                    for kx, ky, kc in obs.keypoints:
                        if kc > 0.25:
                            cv2.circle(vis_frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)

                cv2.rectangle(vis_frame, (10, 10), (460, 48), (20, 20, 20), -1)
                cv2.putText(vis_frame, f"STATE: {state} ({conf*100:.0f}%) | Overlap: {obs.bed_overlap*100:.0f}%", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
                annotated_frames.append(vis_frame)

            sampled_count += 1
            if max_frames and sampled_count >= max_frames:
                break

            # Progress log every 10 seconds of video
            if sampled_count % int(fps * 10) == 0:
                elapsed_proc = time.time() - start_wall_time
                fps_proc = sampled_count / max(1e-3, elapsed_proc)
                logger.info(f"Progress: {format_hms(ts)} / {format_hms(total_dur_sec)} ({ts/max(1e-3, total_dur_sec)*100:3.0f}%) | Processing Speed: {fps_proc:.1f} FPS")

        frame_idx += 1
    cap.release()

    if not raw_records:
        raise RuntimeError("No frames could be processed.")

    obs_duration = raw_records[-1][0] - raw_records[0][0] if len(raw_records) > 1 else 0.0

    # 3. Temporal Processing: Smoothing Pass (§3)
    logger.info("Applying Temporal Smoothing (merging blips < 2.0s)...")
    smoothed_segments = smoother.smooth(raw_records)
    logger.state(f"Generated {len(smoothed_segments)} Temporally Smoothed State Segments:")
    for idx, seg in enumerate(smoothed_segments, 1):
        logger.state(f"  Segment {idx:02d}: [{format_hms(seg.start_t)} -> {format_hms(seg.end_t)}] {seg.state:<18} (Duration: {seg.duration:>4.1f}s, Confidence: {seg.confidence*100:4.1f}%)")


    # 4. State Mechanism: Bed Pattern Matching (§4.2)
    logger.info("Evaluating Bed Pattern Matcher for Bed-Exit and Bed-Return sequences...")
    candidate_events = pattern_matcher.find_candidates(smoothed_segments)

    # 5. State Policy Evaluation (§6.1)
    final_events: List[BedEvent] = []
    for match_ev in candidate_events:
        decision = policy.evaluate_event(match_ev.event_type, out_of_bed_sec=5.0)
        ev_obj = BedEvent(
            event=match_ev.event_type,
            start_time=format_hms(match_ev.start_t),
            confirmed_time=format_hms(match_ev.confirmed_t),
            previous_state=match_ev.prev_state,
            current_state=match_ev.curr_state,
            confidence=match_ev.confidence,
            decision=decision,
            tool_trace=[],
        )
        final_events.append(ev_obj)
        logger.state(
            f"*** BED EVENT DETECTED: {ev_obj.event.upper()} [{ev_obj.decision}] "
            f"at {ev_obj.confirmed_time} | Transition: {ev_obj.previous_state} -> {ev_obj.current_state} "
            f"(Confidence: {ev_obj.confidence*100:.0f}%) ***"
        )

    # 6. JSON Reports Export (§6.2)
    ReportGenerator.export(smoothed_segments, final_events, output_dir, obs_duration)
    summary_report = ReportGenerator.build_summary(smoothed_segments, final_events, obs_duration)
    logger.info(f"JSON reports exported successfully: {output_dir}/(timeline.json, events.json, summary.json)")

    # 7. Annotated Video Export
    if annotated_video and annotated_frames:
        logger.info(f"Encoding annotated video output: {annotated_video}...")
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_video, fourcc, fps, (w, h))
        for f in annotated_frames:
            writer.write(f)
        writer.release()
        logger.info(f"Annotated video saved successfully: {annotated_video}")

    # 8. Print Executive Clinical Monitoring Report
    print("\n" + "=" * 72)
    print("                    BEDSENSE AI CLINICAL MONITORING REPORT")
    print("=" * 72)
    print(f"Observation Duration:      {format_hms(summary_report['observation_duration_sec'])} ({summary_report['observation_duration_sec']}s)")
    print(f"Bed Exit Count:            {summary_report['bed_exit_count']}")
    print(f"Bed Return Count:          {summary_report['bed_return_count']}")
    print(f"Total In-Bed Time:         {format_hms(summary_report['total_in_bed_sec'])} ({summary_report['total_in_bed_sec']}s)")
    print(f"Total Out-of-Bed Time:     {format_hms(summary_report['total_out_of_bed_sec'])} ({summary_report['total_out_of_bed_sec']}s)")
    print(f"Longest Out-of-Bed Period: {format_hms(summary_report['longest_out_of_bed_period_sec'])} ({summary_report['longest_out_of_bed_period_sec']}s)")
    print(f"Final Patient State:       {summary_report['final_state'].upper()}")

    print("\n--- Activity Breakdown (100% Duration Conservation) ---")
    for act, dur in summary_report["activity_duration_sec"].items():
        pct = (dur / max(1, summary_report['observation_duration_sec'])) * 100
        print(f"  • {act.replace('_', ' ').title():<22}: {format_hms(dur)} ({dur:>3}s | {pct:4.1f}%)")

    print("\n--- Confirmed Bed Events ---")
    if final_events:
        for ev in final_events:
            print(f"  [{ev.decision}] {ev.event.upper()} at {ev.confirmed_time} | {ev.previous_state} -> {ev.current_state} (Conf: {ev.confidence*100:.0f}%)")
    else:
        print("  No bed exit or return events detected.")

    print("\n--- Temporally Smoothed Timeline Segments ---")
    for s in smoothed_segments:
        print(f"  [{format_hms(s.start_t)} -> {format_hms(s.end_t)}] {s.state:<18} (Duration: {s.duration:>4.1f}s, Confidence: {s.confidence:.2f})")

    print("=" * 72)
    print(f"Artifacts and reports saved to: {Path(output_dir).resolve()}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BedSense AI — Vision -> Temporal Processing -> State Mechanism CLI"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for JSON reports")
    parser.add_argument("--annotated-video", type=str, default=None, help="Optional annotated video MP4 output path")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS sampling rate (default: 10.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--min-segment-sec", type=float, default=2.0, help="Minimum segment smoothing duration in seconds (default: 2.0)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda:0 or cpu)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "STATE_ONLY", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    process_video_cli(
        video_path=args.video,
        output_dir=args.output_dir,
        annotated_video=args.annotated_video,
        fps=args.fps,
        max_frames=args.max_frames,
        min_segment_sec=args.min_segment_sec,
        device=args.device,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
