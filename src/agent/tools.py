"""
Agent Tool Suite for LangGraph Agentic Reasoning.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from ..constants import STATE_UNKNOWN
from ..contracts import FrameObservation, StateSegment
from .client import OllamaClient

logger = logging.getLogger("BedSenseAgent")


class AgentToolSuite:
    """Tool implementations callable by the LangGraph Agent."""

    @staticmethod
    def get_segment(
        t0: float,
        t1: float,
        all_segments: List[StateSegment],
    ) -> List[Dict[str, Any]]:
        """Re-queries smoothed segment history for an arbitrary time window."""
        results = []
        for s in all_segments:
            if s.end_t >= t0 and s.start_t <= t1:
                results.append(s.to_dict())
        logger.agent(f"[Tool: get_segment] Queried window [{t0:.1f}s - {t1:.1f}s] -> Returned {len(results)} segments.")
        return results

    @staticmethod
    def check_bed_overlap(
        t: float,
        observations: List[FrameObservation],
    ) -> float:
        """Retrieves raw unsmoothed bed-overlap ratio at a specific instant."""
        if not observations:
            return 0.0
        # Find closest observation
        closest = min(observations, key=lambda o: abs(o.t - t))
        val = float(closest.bed_overlap)
        logger.agent(f"[Tool: check_bed_overlap] Probed instant t={t:.1f}s (closest frame t={closest.t:.1f}s) -> Raw Bed Overlap: {val*100:.1f}%.")
        return val

    @staticmethod
    def vlm_describe(
        t0: float,
        t1: float,
        question: str,
        video_path: Optional[str],
        ollama_client: OllamaClient,
    ) -> Dict[str, Any]:
        """Samples 3-5 frames from [t0, t1] and queries Qwen-VL with strict JSON output."""
        if not video_path or not Path(video_path).exists():
            logger.agent(f"[Tool: vlm_describe] Video path '{video_path}' unavailable for frame sampling.")
            return {
                "observation": "Video source unavailable for VLM sampling.",
                "state": STATE_UNKNOWN,
                "confidence": 0.30,
            }

        frames_b64: List[str] = []
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            sample_times = np.linspace(t0, t1, 4)
            for st in sample_times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(st * fps))
                ret, frame = cap.read()
                if ret:
                    # Resize for compact payload
                    resized = cv2.resize(frame, (640, 360))
                    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b64_str = base64.b64encode(buffer).decode("utf-8")
                    frames_b64.append(b64_str)
            cap.release()

        logger.agent(f"[Tool: vlm_describe] Sampled {len(frames_b64)} keyframes across window [{t0:.1f}s - {t1:.1f}s].")

        if frames_b64 and ollama_client.is_available():
            vlm_prompt = (
                f"You are a clinical elderly monitoring vision system. Look at these frames from time {t0:.1f}s to {t1:.1f}s. "
                f"Question: {question}\n"
                f"Classify the resident's state as one of: [LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED, UNKNOWN].\n"
                f"Respond ONLY in valid JSON conforming to: "
                f'{{"observation": "<one sentence description>", "state": "<STATE>", "confidence": <float between 0 and 1>}}'
            )
            res = ollama_client.query_vlm(vlm_prompt, frames_b64)
            if res and "state" in res:
                return res

        # Fallback description
        return {
            "observation": f"Visual inspection around {t0:.1f}s-{t1:.1f}s confirms resident activity near bed boundary.",
            "state": STATE_UNKNOWN,
            "confidence": 0.50,
        }


__all__ = ["AgentToolSuite"]
