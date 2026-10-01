"""
Command-Line Interface for BedSense AI Monitoring System.
Executes the unified pipeline: Vision -> Temporal Processing -> State Mechanism.

Provides multi-level logging with high-visibility state transitions & bed events.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from .agent import LangGraphAgentWorkflow, OllamaClient
from .constants import STATE_COLORS
from .logging_config import load_config, logger, setup_logger
from .policy import PolicyEngine, ReportGenerator, format_hms
from .state import BedPatternMatcher
from .streaming import StreamingProcessor
from .temporal import FrameClassifier
from .vision import BedRelationEngine, PerceptionDetector

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Video Processing Pipeline
# ---------------------------------------------------------------------------

def process_video_cli(
    video_path: str,
    output_dir: str = "outputs",
    annotated_video: Optional[str] = None,
    config_path: Optional[str] = "configs/configurations.yaml",
    fps: float = 10.0,
    max_frames: Optional[int] = None,
    min_segment_sec: float = 2.0,
    device: Optional[str] = None,
    log_level: Optional[str] = None,
    enable_agent: bool = True,
    ollama_url: Optional[str] = None,
    text_model: Optional[str] = None,
    vlm_model: Optional[str] = None,
    detector_model: Optional[str] = None,
    pose_model: Optional[str] = None,
):
    cfg = load_config(config_path)
    setup_logger(log_level, config=cfg)
    perception_cfg = cfg.get("perception", {}) if isinstance(cfg, dict) else {}
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}

    if not detector_model:
        detector_model = perception_cfg.get("detector_model", "yolo11n.pt")
    if not pose_model:
        pose_model = perception_cfg.get("pose_model", "yolo11n-pose.pt")
    pose_conf = float(perception_cfg.get("pose_conf", 0.20))
    bed_conf = float(perception_cfg.get("bed_conf", 0.12))

    if not ollama_url:
        ollama_url = os.getenv("OLLAMA_BASE_URL", agent_cfg.get("ollama_base_url", "http://ollama:11434"))
    if not text_model:
        text_model = agent_cfg.get("text_model", "qwen2.5:3b")
    if not vlm_model:
        vlm_model = agent_cfg.get("vlm_model", "qwen2.5vl:3b")

    # Select compute device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using compute device: {device}")
    logger.info(f"Initializing Vision Layer (Detector: {detector_model}, Pose: {pose_model})...")

    bed_engine = BedRelationEngine(
        model_path=detector_model,
        confidence_thresh=bed_conf,
        device=device,
    )
    detector = PerceptionDetector(
        model_path=pose_model,
        conf_threshold=pose_conf,
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

    # 2. Pipeline Engines & Real-Time Streaming State Engine
    classifier = FrameClassifier()
    pattern_matcher = BedPatternMatcher(min_exit_confirm_sec=3.0, return_window_sec=30.0)
    policy = PolicyEngine(
        edge_sit_monitor_sec=10.0,
        unknown_monitor_sec=10.0,
        out_of_bed_alert_sec=10.0,
        suppress_with_caregiver=True,
    )
    agent_workflow = LangGraphAgentWorkflow(
        ollama_client=OllamaClient(
            base_url=ollama_url,
            text_model=text_model,
            vlm_model=vlm_model,
        )
    )

    streaming_processor = StreamingProcessor(
        min_segment_sec=min_segment_sec,
        context_window_sec=5.0,
        pattern_matcher=pattern_matcher,
        policy_engine=policy,
        agent_workflow=agent_workflow,
        video_path=video_path,
        enable_agent=enable_agent,
    )

    annotated_frames: List[np.ndarray] = []
    telemetry_records: List[Dict[str, Any]] = []
    frame_idx = 0
    sampled_count = 0
    start_wall_time = time.time()

    logger.info("Starting Real-Time Live Streaming Pipeline: Vision -> Online Temporal Processing -> On-the-Fly Agent Reasoning...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts = round(frame_idx / native_fps, 2)

            # 1. Vision Detection & Tracking
            obs = detector.get_primary_observation(frame, ts)

            # 2. Instantaneous Posture Classification
            state, conf = classifier.classify(obs, bed_engine=bed_engine)

            # 3. Real-Time Streaming State Engine (On-the-fly temporal tracking, agent reasoning, and live alerts)
            live_status = streaming_processor.process_frame(ts, state, conf, obs)
            live_state = live_status["live_state"]
            live_conf = live_status["live_conf"]

            # 4. Telemetry Record for CSV Export (Bed geometry, Person BBox, Keypoints, Biomechanics, and State)
            metrics = classifier.get_last_metrics()
            bed_poly = bed_engine.polygon.tolist() if bed_engine.polygon is not None else None
            bed_bbox = bed_engine.bbox

            record: Dict[str, Any] = {
                "frame_idx": frame_idx,
                "timestamp_sec": ts,
                "time_hms": format_hms(ts),
                "track_id": obs.track_id if obs.track_id is not None else "",
                "person_bbox_x1": round(obs.bbox[0], 2) if obs.bbox else "",
                "person_bbox_y1": round(obs.bbox[1], 2) if obs.bbox else "",
                "person_bbox_x2": round(obs.bbox[2], 2) if obs.bbox else "",
                "person_bbox_y2": round(obs.bbox[3], 2) if obs.bbox else "",
                "person_bbox_w": round(obs.bbox[2] - obs.bbox[0], 2) if obs.bbox else "",
                "person_bbox_h": round(obs.bbox[3] - obs.bbox[1], 2) if obs.bbox else "",
                "bbox_aspect_ratio": metrics.get("aspect_ratio", "") if metrics.get("aspect_ratio") is not None else "",
                "bed_overlap": round(obs.bed_overlap, 3),
                "mean_kpt_conf": round(obs.mean_kpt_conf, 3),
                "torso_angle_deg": metrics.get("torso_angle", "") if metrics.get("torso_angle") is not None else "",
                "knee_angle_deg": metrics.get("knee_angle", "") if metrics.get("knee_angle") is not None else "",
                "velocity_px_sec": metrics.get("velocity", 0.0),
                "is_aligned_with_bed": metrics.get("is_aligned_with_bed_plane", False),
                "instantaneous_state": state,
                "instantaneous_conf": round(conf, 3),
                "smoothed_state": live_state,
                "smoothed_conf": round(live_conf, 3),
                "active_event": live_status.get("active_event_banner") or "",
                "bed_bbox_x1": round(bed_bbox[0], 2) if bed_bbox else "",
                "bed_bbox_y1": round(bed_bbox[1], 2) if bed_bbox else "",
                "bed_bbox_x2": round(bed_bbox[2], 2) if bed_bbox else "",
                "bed_bbox_y2": round(bed_bbox[3], 2) if bed_bbox else "",
                "bed_polygon": json.dumps(bed_poly) if bed_poly is not None else "",
            }

            if obs.keypoints is not None and len(obs.keypoints) == 17:
                for kidx, kname in enumerate(COCO_KEYPOINT_NAMES):
                    kx, ky, kc = obs.keypoints[kidx]
                    record[f"{kname}_x"] = round(float(kx), 2)
                    record[f"{kname}_y"] = round(float(ky), 2)
                    record[f"{kname}_conf"] = round(float(kc), 3)
                record["keypoints_json"] = json.dumps([[round(float(x), 2), round(float(y), 2), round(float(c), 3)] for x, y, c in obs.keypoints])
            else:
                for kname in COCO_KEYPOINT_NAMES:
                    record[f"{kname}_x"] = ""
                    record[f"{kname}_y"] = ""
                    record[f"{kname}_conf"] = ""
                record["keypoints_json"] = ""

            telemetry_records.append(record)

            # Debug Telemetry Log
            logger.debug(
                f"[t={ts:06.2f}s] Live State: {live_state:<18} (Inst: {state:<18}) | Conf: {live_conf*100:4.1f}% | "
                f"Track: {str(obs.track_id):<4} | Overlap: {obs.bed_overlap*100:4.1f}% | "
                f"KptConf: {obs.mean_kpt_conf:4.2f}"
            )

            # Annotated Visual Frame Generation with Live HUD Banner
            if annotated_video:
                vis_frame = frame.copy()
                if bed_engine.polygon is not None:
                    pts = bed_engine.polygon.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_frame, [pts], True, (0, 200, 0), 2)

                if obs.bbox is not None:
                    rx1, ry1, rx2, ry2 = [int(v) for v in obs.bbox]
                    color = STATE_COLORS.get(live_state, (255, 255, 255))
                    cv2.rectangle(vis_frame, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(vis_frame, f"{live_state} (ID:{obs.track_id})", (rx1, max(18, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if obs.keypoints is not None:
                    for kx, ky, kc in obs.keypoints:
                        if kc > 0.25:
                            cv2.circle(vis_frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)

                # Real-Time Live HUD Overlay
                hud_text = f"LIVE: {live_state} ({live_conf*100:.0f}%) | OVERLAP: {obs.bed_overlap*100:.0f}%"
                if live_status.get("active_event_banner"):
                    hud_text += f" | {live_status['active_event_banner']}"

                cv2.rectangle(vis_frame, (10, 10), (620, 48), (20, 20, 20), -1)
                cv2.putText(vis_frame, hud_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
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

    if not streaming_processor.raw_records:
        raise RuntimeError("No frames could be processed.")

    actual_duration = round(frame_idx / native_fps, 2) if frame_idx > 0 else total_dur_sec
    obs_duration = actual_duration

    # Finalize Streaming State Segments & Events with 100% Duration Conservation
    smoothed_segments, final_events = streaming_processor.finalize(actual_duration)

    logger.state(f"Completed Stream Ingestion. Generated {len(smoothed_segments)} Continuous Timeline State Segments:")
    for idx, seg in enumerate(smoothed_segments, 1):
        logger.state(f"  Segment {idx:02d}: [{format_hms(seg.start_t)} -> {format_hms(seg.end_t)}] {seg.state:<18} (Duration: {seg.duration:>4.1f}s, Confidence: {seg.confidence*100:4.1f}%)")

    # 6. JSON Reports & CSV Telemetry Export (§6.2)
    ReportGenerator.export(smoothed_segments, final_events, output_dir, obs_duration, telemetry_records=telemetry_records)
    summary_report = ReportGenerator.build_summary(smoothed_segments, final_events, obs_duration)
    logger.info(f"Reports exported successfully to '{output_dir}/': timeline.json, events.json, summary.json, telemetry.csv ({len(telemetry_records)} rows)")

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

    # 8. Print Executive Clinical Monitoring Report & Complete Summary JSON
    summary_json_str = json.dumps(summary_report, indent=2)

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print("\n" + f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║         BEDSENSE AI CLINICAL MONITORING COMPLETE SUMMARY                 ║{RESET}")
    print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{summary_json_str}")
    print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    print("=" * 72)
    print("                    BEDSENSE AI CLINICAL MONITORING REPORT")
    print("=" * 72)
    print(f"Observation Duration:      {format_hms(summary_report['observation_duration_sec'])} ({summary_report['observation_duration_sec']}s)")
    print(f"Bed Exit Count:            {summary_report['bed_exit_count']}")
    print(f"Bed Return Count:          {summary_report['bed_return_count']}")
    if "missing_count" in summary_report:
        print(f"Missing (Alert) Count:     {summary_report['missing_count']}")
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
        description="BedSense AI — Vision -> Temporal Processing -> LangGraph Agent -> State Mechanism CLI"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for JSON reports")
    parser.add_argument("--annotated-video", type=str, default=None, help="Optional annotated video MP4 output path")
    parser.add_argument("--config", type=str, default="configs/configurations.yaml", help="Path to YAML configuration file (default: configs/configurations.yaml)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS sampling rate (default: 10.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--min-segment-sec", type=float, default=2.0, help="Minimum segment smoothing duration in seconds (default: 2.0)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda:0 or cpu)")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["PROD", "INFO", "AGENT", "DEBUG", "STATE", "WARNING", "ERROR"],
        help="Logging verbosity level (default: read from configs/configurations.yaml or PROD)",
    )
    parser.add_argument("--enable-agent", action="store_true", default=True, help="Enable LangGraph Agent layer for ambiguous events (default: True)")
    parser.add_argument("--no-agent", dest="enable_agent", action="store_false", help="Disable LangGraph Agent layer (heuristic only)")
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama server URL (default: read from configs/configurations.yaml or http://ollama:11434)",
    )
    parser.add_argument("--text-model", type=str, default=None, help="Ollama text model (default: read from config or qwen2.5:3b)")
    parser.add_argument("--vlm-model", type=str, default=None, help="Ollama VLM model (default: read from config or qwen2.5vl:3b)")
    parser.add_argument("--detector-model", type=str, default=None, help="YOLO object detection model path (default: read from config or yolo11n.pt)")
    parser.add_argument("--pose-model", type=str, default=None, help="YOLO pose estimation model path (default: read from config or yolo11n-pose.pt)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    process_video_cli(
        video_path=args.video,
        output_dir=args.output_dir,
        annotated_video=args.annotated_video,
        config_path=args.config,
        fps=args.fps,
        max_frames=args.max_frames,
        min_segment_sec=args.min_segment_sec,
        device=args.device,
        log_level=args.log_level,
        enable_agent=args.enable_agent,
        ollama_url=args.ollama_url,
        text_model=args.text_model,
        vlm_model=args.vlm_model,
        detector_model=args.detector_model,
        pose_model=args.pose_model,
    )


if __name__ == "__main__":
    main()
