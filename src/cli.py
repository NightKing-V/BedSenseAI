"""
Command-Line Interface for BedSense AI Monitoring System.
Executes the unified evidence-based pipeline: Vision -> Biomechanical Evidence Engine -> State Mechanism.

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
from .evidence import EvidenceTemporalClassifier, RuleBasedEvidenceExtractor
from .exporter import ArtifactExporter, VideoAnnotator
from .logging_config import load_config, logger, setup_logger
from .policy import PolicyEngine, format_hms
from .state import BedPatternMatcher
from .streaming import StreamingProcessor
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
    save_videos: bool = True,
    annotated_video: Optional[str] = None,
    config_path: Optional[str] = "configs/configurations.yaml",
    fps: float = 10.0,
    max_frames: Optional[int] = None,
    min_segment_sec: Optional[float] = None,
    device: Optional[str] = None,
    log_level: Optional[str] = None,
    enable_agent: bool = False,
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

    temporal_cfg = cfg.get("temporal", {}) if isinstance(cfg, dict) else {}
    if min_segment_sec is None:
        min_segment_sec = float(temporal_cfg.get("min_segment_sec", 0.5))
    min_exit_confirm_sec = float(temporal_cfg.get("min_exit_confirm_sec", 3.0))
    return_window_sec = float(temporal_cfg.get("return_window_sec", 10.0))
    context_window_sec = float(temporal_cfg.get("context_window_sec", 5.0))

    # 2. Pipeline Engine: Evidence-Based Biomechanical Engine
    evidence_cfg = cfg.get("evidence", {}) if isinstance(cfg, dict) else {}
    evidence_extractor = RuleBasedEvidenceExtractor(
        low_body_motion=float(evidence_cfg.get("low_body_motion", 0.015)),
        high_body_motion=float(evidence_cfg.get("high_body_motion", 0.035)),
        extended_knee_angle=float(evidence_cfg.get("extended_knee_angle", 160.0)),
        bent_knee_angle=float(evidence_cfg.get("bent_knee_angle", 130.0)),
        extended_hip_angle=float(evidence_cfg.get("extended_hip_angle", 160.0)),
        bent_hip_angle=float(evidence_cfg.get("bent_hip_angle", 130.0)),
        on_bed_threshold=float(evidence_cfg.get("on_bed_threshold", 0.25)),
        off_bed_threshold=float(evidence_cfg.get("off_bed_threshold", 0.12)),
        torso_length_normalized_threshold=float(evidence_cfg.get("torso_length_normalized_threshold", 1.25)),
    )
    classifier_evidence = EvidenceTemporalClassifier(
        min_keypoint_conf=float(evidence_cfg.get("min_keypoint_conf", 0.25)),
        window_size=int(evidence_cfg.get("window_size", 5)),
        minimum_confidence=float(evidence_cfg.get("minimum_confidence", 0.30)),
        evidence_extractor=evidence_extractor,
    )
    pattern_matcher = BedPatternMatcher(min_exit_confirm_sec=min_exit_confirm_sec, return_window_sec=return_window_sec)

    policy_cfg = cfg.get("policy", {}) if isinstance(cfg, dict) else {}
    policy_engine = PolicyEngine(
        edge_sit_monitor_sec=float(policy_cfg.get("edge_sit_monitor_sec", 10.0)),
        unknown_monitor_sec=float(policy_cfg.get("unknown_monitor_sec", 10.0)),
        out_of_bed_alert_sec=float(policy_cfg.get("out_of_bed_alert_sec", 10.0)),
        caregiver_suppress_escalation=bool(policy_cfg.get("caregiver_suppress_escalation", True)),
    )

    agent_workflow = None
    if enable_agent:
        agent_workflow = LangGraphAgentWorkflow(
            ollama_client=OllamaClient(
                base_url=ollama_url,
                text_model=text_model,
                vlm_model=vlm_model,
            )
        )
        logger.info("LangGraph Agent reasoning layer ENABLED for ambiguous transitions.")
    else:
        logger.info("Agent layer BYPASSED (evaluating direct temporal biomechanical state outputs).")

    streaming = StreamingProcessor(
        min_segment_sec=min_segment_sec,
        context_window_sec=context_window_sec,
        pattern_matcher=pattern_matcher,
        policy_engine=policy_engine,
        agent_workflow=agent_workflow,
        video_path=video_path,
        enable_agent=enable_agent,
    )

    annotated_frames: List[np.ndarray] = []
    telemetry_records: List[Dict[str, Any]] = []
    frame_idx = 0
    sampled_count = 0
    start_wall_time = time.time()

    should_render_videos = save_videos or (annotated_video is not None)
    logger.info(f"Pipeline Active: Evidence-Based Biomechanics (Video Rendering: {'ENABLED' if should_render_videos else 'DISABLED'})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts = round(frame_idx / native_fps, 2)
            frame_h, frame_w = frame.shape[:2]

            # 1. Vision Detection & Tracking
            obs = detector.get_primary_observation(frame, ts)

            # 2. Evidence-Based Biomechanical Classification
            state, conf = classifier_evidence.classify(
                obs,
                bed_engine=bed_engine,
                frame_width=float(frame_w),
                frame_height=float(frame_h),
            )
            live_result = streaming.process_frame(ts, state, conf, obs)
            metrics = classifier_evidence.get_last_metrics()

            bed_poly = bed_engine.polygon.tolist() if bed_engine.polygon is not None else None
            bed_bbox = bed_engine.bbox

            # 3. Telemetry Record for CSV Export
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
                "bed_overlap": round(obs.bed_overlap, 3),
                "mean_kpt_conf": round(obs.mean_kpt_conf, 3),
                "instantaneous_state": state,
                "instantaneous_conf": round(conf, 3),
                "smoothed_state": live_result["live_state"],
                "smoothed_conf": round(live_result["live_conf"], 3),
                "evidence_lying": metrics.get("evidence_lying", 0.0),
                "evidence_sitting": metrics.get("evidence_sitting", 0.0),
                "evidence_standing": metrics.get("evidence_standing", 0.0),
                "evidence_walking": metrics.get("evidence_walking", 0.0),
                "mean_bed_affinity": metrics.get("mean_bed_affinity", 0.0),
                "torso_length_normalized": metrics.get("torso_length_normalized", ""),
                "knee_angle_deg": metrics.get("knee_angle_deg", ""),
                "hip_angle_deg": metrics.get("hip_angle_deg", ""),
                "body_center_velocity": metrics.get("body_center_velocity", 0.0),
                "ankle_velocity": metrics.get("ankle_velocity", 0.0),
                "knee_velocity": metrics.get("knee_velocity", 0.0),
            }

            # Bed Geometry
            record["bed_bbox_x1"] = round(bed_bbox[0], 2) if bed_bbox else ""
            record["bed_bbox_y1"] = round(bed_bbox[1], 2) if bed_bbox else ""
            record["bed_bbox_x2"] = round(bed_bbox[2], 2) if bed_bbox else ""
            record["bed_bbox_y2"] = round(bed_bbox[3], 2) if bed_bbox else ""
            record["bed_polygon"] = json.dumps(bed_poly) if bed_poly is not None else ""

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

            logger.debug(
                f"[t={ts:06.2f}s] State: {live_result['live_state']:<16} ({live_result['live_conf']*100:3.0f}%) | "
                f"Affinity: {metrics.get('mean_bed_affinity', 0):.2f} | Overlap: {obs.bed_overlap*100:4.1f}%"
            )

            # 4. Annotated Frame Rendering
            if should_render_videos:
                annotated_frame = VideoAnnotator.render_frame_evidence(frame, obs, bed_engine, live_result, metrics)
                annotated_frames.append(annotated_frame)

            sampled_count += 1
            if max_frames and sampled_count >= max_frames:
                break

            # Progress log every 10 seconds of video
            if sampled_count % int(fps * 10) == 0:
                elapsed_proc = time.time() - start_wall_time
                fps_proc = sampled_count / max(1e-3, elapsed_proc)
                logger.info(f"Progress: {format_hms(ts)} / {format_hms(total_dur_sec)} ({ts/max(1e-3, total_dur_sec)*100:3.0f}%) | Speed: {fps_proc:.1f} FPS")

        frame_idx += 1
    cap.release()

    if not streaming.raw_records:
        raise RuntimeError("No frames could be processed.")

    actual_duration = round(frame_idx / native_fps, 2) if frame_idx > 0 else total_dur_sec
    obs_duration = actual_duration

    # Finalize Streaming State Segments & Events
    segs, events = streaming.finalize(actual_duration)

    logger.state(f"Completed Stream Ingestion. Generated {len(segs)} Segments:")
    for idx, seg in enumerate(segs, 1):
        logger.state(f"  Seg {idx:02d}: [{format_hms(seg.start_t)} -> {format_hms(seg.end_t)}] {seg.state:<20} (Duration: {seg.duration:>4.1f}s, Conf: {seg.confidence*100:4.1f}%)")

    # 5. Export Primary Artifacts
    video_out_name = Path(annotated_video).name if annotated_video else "annotated.mp4"
    ArtifactExporter.export_primary(
        output_dir=output_dir,
        segments=segs,
        events=events,
        obs_duration=obs_duration,
        telemetry_records=telemetry_records,
        frames=annotated_frames if should_render_videos else None,
        video_name=video_out_name,
        fps=fps,
    )
    ArtifactExporter.print_terminal_summary(
        obs_duration=obs_duration,
        segments=segs,
        events=events,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BedSense AI — Clinical Vision & Evidence-Based State Monitoring Pipeline"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports and video")
    parser.add_argument("--save-videos", action="store_true", default=True, help="Export annotated MP4 video (default: True)")
    parser.add_argument("--no-videos", dest="save_videos", action="store_false", help="Disable annotated MP4 video rendering for maximum speed")
    parser.add_argument("--annotated-video", type=str, default=None, help="Optional custom annotated video output file path")
    parser.add_argument("--config", type=str, default="configs/configurations.yaml", help="Path to YAML configuration file")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS sampling rate (default: 10.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--min-segment-sec", type=float, default=None, help="Minimum segment smoothing duration in seconds (default: read from config or 0.5s)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda:0 or cpu)")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["PROD", "INFO", "AGENT", "DEBUG", "STATE", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    parser.add_argument("--enable-agent", action="store_true", default=False, help="Enable LangGraph Agent layer (default: False)")
    parser.add_argument("--no-agent", dest="enable_agent", action="store_false", help="Disable LangGraph Agent layer")
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
        save_videos=args.save_videos,
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
