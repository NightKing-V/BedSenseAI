"""
Real-Time Online Streaming Processor for BedSense AI (§3, §4, §5, §6 of Design Spec).
Processes live video/camera frames on-the-fly, updates temporal state segments in real-time,
triggers LangGraph agent reasoning immediately upon ambiguous transitions, and emits live clinical alerts.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..agent import LangGraphAgentWorkflow
from ..constants import (
    DECISION_ALERT,
    DECISION_MONITOR,
    DECISION_NORMAL,
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    EVENT_MISSING,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_UNKNOWN,
)
from ..contracts import BedEvent, FrameObservation, StateSegment
from ..policy import PolicyEngine, format_hms
from ..state import BedPatternMatcher, CandidateEvent
from ..temporal import evaluate_ambiguity

AGENT_LEVEL_NUM = 15
STATE_LEVEL_NUM = 25
if not hasattr(logging, "AGENT"):
    logging.addLevelName(AGENT_LEVEL_NUM, "AGENT")
if not hasattr(logging, "STATE"):
    logging.addLevelName(STATE_LEVEL_NUM, "STATE")

if not hasattr(logging.Logger, "agent"):
    def log_agent(self, message, *args, **kws):
        if self.isEnabledFor(AGENT_LEVEL_NUM):
            self._log(AGENT_LEVEL_NUM, message, args, **kws)
    logging.Logger.agent = log_agent

if not hasattr(logging.Logger, "state"):
    def log_state(self, message, *args, **kws):
        if self.isEnabledFor(STATE_LEVEL_NUM):
            self._log(STATE_LEVEL_NUM, message, args, **kws)
    logging.Logger.state = log_state

logger = logging.getLogger("BedSenseStreaming")


class StreamingProcessor:
    """
    Online Streaming State Engine that processes frame observations frame-by-frame on-the-fly.
    """

    def __init__(
        self,
        min_segment_sec: float = 1.0,
        context_window_sec: float = 5.0,
        pattern_matcher: Optional[BedPatternMatcher] = None,
        policy_engine: Optional[PolicyEngine] = None,
        agent_workflow: Optional[LangGraphAgentWorkflow] = None,
        video_path: Optional[str] = None,
        enable_agent: bool = True,
    ):
        self.min_segment_sec = min_segment_sec
        self.context_window_sec = context_window_sec
        self.pattern_matcher = pattern_matcher or BedPatternMatcher()
        self.policy_engine = policy_engine or PolicyEngine()
        self.agent_workflow = agent_workflow
        self.video_path = video_path
        self.enable_agent = enable_agent

        # Stream history buffers
        self.observations: List[FrameObservation] = []
        self.raw_records: List[Tuple[float, str, float, FrameObservation]] = []
        self.finalized_segments: List[StateSegment] = []
        self.final_events: List[BedEvent] = []
        self.processed_event_keys: Set[Tuple[str, float]] = set()

        # Active streaming state
        self.active_state: Optional[str] = None
        self.active_start_t: float = 0.0
        self.active_confs: List[float] = []

        # Pending state change (for online blip absorption)
        self.pending_state: Optional[str] = None
        self.pending_start_t: float = 0.0
        self.pending_confs: List[float] = []

        # Live overlay tracking
        self.last_event_banner: Optional[str] = None
        self.last_event_banner_time: float = -999.0

    def process_frame(
        self,
        t: float,
        state: str,
        conf: float,
        obs: FrameObservation,
    ) -> Dict[str, Any]:
        """
        Ingests a single frame observation on-the-fly, updates streaming temporal states,
        and triggers online agent reasoning and alerts when state transitions occur.
        """
        self.observations.append(obs)
        self.raw_records.append((t, state, conf, obs))

        # 1. Stream Initialization (First frame)
        if self.active_state is None:
            self.active_state = state
            self.active_start_t = t
            self.active_confs = [conf]
            logger.state(f"[STREAM START] Initial Resident State at {format_hms(t)}: {state} (Confidence: {conf*100:.0f}%)")
            return self._build_live_status(t, state, conf, obs)

        in_bed_states = {STATE_LYING_IN_BED, STATE_SITTING_ON_BED}

        # 2. State Continuity or Blip Return
        if state == self.active_state:
            # If a transient state was pending, but reverted before threshold, absorb the blip
            if self.pending_state is not None:
                self.pending_state = None
                self.pending_confs.clear()
            self.active_confs.append(conf)

        # 3. Candidate State Transition
        else:
            # Determine threshold: intra-bed switches (lying <-> sitting on bed) use fast 0.35s confirmation
            if self.active_state in in_bed_states and state in in_bed_states:
                required_duration = min(0.35, self.min_segment_sec)
            else:
                required_duration = self.min_segment_sec

            # Fast In-Bed Handoff: if pending_state was already in-bed, and new state is also in-bed, preserve pending_start_t!
            is_in_bed_handoff = (
                self.pending_state in in_bed_states
                and state in in_bed_states
            )

            if self.pending_state != state and not is_in_bed_handoff:
                # Start timing a new candidate state
                self.pending_state = state
                self.pending_start_t = t
                self.pending_confs = [conf]
            else:
                # In-bed handoff updates the pending state name while preserving entry start time
                if is_in_bed_handoff and self.pending_state != state:
                    self.pending_state = state
                self.pending_confs.append(conf)

                # Check if pending state has persisted long enough to confirm a valid transition (§3)
                if (t - self.pending_start_t) >= required_duration:
                    self._commit_transition(t)

        # 4. Check for real-time live events (e.g. missing threshold >= 10s or out_of_bed -> unknown)
        if self.active_state not in in_bed_states:
            self._evaluate_live_events(t)

        return self._build_live_status(t, state, conf, obs)

    def _commit_transition(self, current_t: float):
        """Commits the completed previous state segment and triggers real-time pattern and agent evaluation."""
        prev_conf = float(np.mean(self.active_confs)) if self.active_confs else 0.50
        committed_seg = StateSegment(
            start_t=self.active_start_t,
            end_t=self.pending_start_t,
            state=self.active_state,
            confidence=prev_conf,
        )
        self.finalized_segments.append(committed_seg)

        logger.state(
            f"[LIVE STATE TRANSITION] [{format_hms(committed_seg.start_t)} -> {format_hms(committed_seg.end_t)}] "
            f"{committed_seg.state} (Duration: {committed_seg.duration:.1f}s, Conf: {committed_seg.confidence*100:.0f}%) -> {self.pending_state}"
        )

        # Switch active state to the new confirmed state
        self.active_state = self.pending_state
        self.active_start_t = self.pending_start_t
        self.active_confs = list(self.pending_confs)
        self.pending_state = None
        self.pending_confs.clear()

        # Real-time event matching on current segment history
        self._evaluate_live_events(current_t)

    def _evaluate_live_events(self, current_t: float):
        """Evaluates BedPatternMatcher and LangGraph Agent on-the-fly for newly completed transitions and threshold alerts."""
        current_active_seg = StateSegment(
            start_t=self.active_start_t,
            end_t=current_t,
            state=self.active_state,
            confidence=float(np.mean(self.active_confs)) if self.active_confs else 0.50,
        )
        working_segments = self.finalized_segments + [current_active_seg]

        candidates = self.pattern_matcher.find_candidates(working_segments)
        for cand in candidates:
            event_key = (cand.event_type, round(cand.start_t, 2))
            if event_key in self.processed_event_keys:
                continue
            self.processed_event_keys.add(event_key)

            # Find matching segment for context
            matching_seg = next(
                (s for s in working_segments if s.start_t <= cand.start_t <= s.end_t),
                working_segments[-1]
            )

            # Evaluate ambiguity on-the-fly (§4.1)
            seg_obs = [o for o in self.observations if matching_seg.start_t <= o.t <= matching_seg.end_t]
            is_amb, amb_conf, amb_reason = evaluate_ambiguity(matching_seg, seg_obs, [cand])
            if is_amb:
                matching_seg.confidence = amb_conf
                matching_seg.ambiguous_reason = amb_reason

            conf = cand.confidence
            out_dur = max(0.0, current_t - cand.start_t) if cand.event_type in (EVENT_BED_EXIT, EVENT_MISSING) else 0.0
            decision = self.policy_engine.evaluate_event(cand.event_type, out_of_bed_sec=out_dur)
            tool_trace: List[Dict[str, Any]] = []

            # Invoke LangGraph Agent if ambiguous or agent enabled
            should_query_agent = (
                cand.is_ambiguous
                or is_amb
                or (matching_seg.ambiguous_reason is not None)
                or self.enable_agent
            )

            if should_query_agent and self.agent_workflow is not None:
                trigger_reason = cand.ambiguous_reason or matching_seg.ambiguous_reason or "Live Stream Verification"
                logger.agent(
                    f"[LIVE AGENT ROUTE] Routing event '{cand.event_type}' at {format_hms(cand.start_t)} "
                    f"to LangGraph Agent on-the-fly (Trigger: {trigger_reason})..."
                )
                agent_result = self.agent_workflow.run(
                    segment=matching_seg,
                    context_history=working_segments,
                    observations=self.observations,
                    video_path=self.video_path,
                    max_tool_calls=5,
                )
                tool_trace = agent_result.get("tool_trace", [])
                agent_conf = agent_result.get("confidence_score") if agent_result.get("confidence_score") is not None else agent_result.get("final_confidence")
                if agent_conf is not None:
                    conf = float(agent_conf)
                    matching_seg.confidence = conf
                if agent_result.get("reasoning"):
                    logger.agent(f"[LIVE AGENT RESOLUTION] Reasoning: {agent_result['reasoning']} (Confidence: {conf*100:.0f}%)")

            ev_obj = BedEvent(
                event=cand.event_type,
                start_time=format_hms(cand.start_t),
                confirmed_time=format_hms(cand.confirmed_t),
                previous_state=cand.prev_state,
                current_state=cand.curr_state,
                confidence=conf,
                decision=decision,
                tool_trace=tool_trace,
            )
            self.final_events.append(ev_obj)

            self.last_event_banner = f"{ev_obj.event.upper()} [{ev_obj.decision}] ({ev_obj.confidence*100:.0f}%)"
            self.last_event_banner_time = current_t

            event_dict = {
                "event": ev_obj.event,
                "start_time": ev_obj.start_time,
                "confirmed_time": ev_obj.confirmed_time,
                "previous_state": ev_obj.previous_state,
                "current_state": ev_obj.current_state,
                "confidence": round(float(ev_obj.confidence), 2),
                "decision": ev_obj.decision,
            }
            ev_json_str = json.dumps(event_dict, indent=2)

            RED = "\033[91m"
            YELLOW = "\033[93m"
            GREEN = "\033[92m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            if ev_obj.decision == "ALERT":
                c = RED
                tag = "🚨 [LIVE CLINICAL ALERT TRIGGERED] 🚨"
            elif ev_obj.decision == "MONITOR":
                c = YELLOW
                tag = "⚠️ [LIVE CLINICAL MONITORING EVENT] ⚠️"
            else:
                c = GREEN
                tag = "🔔 [LIVE BED EVENT CONFIRMED] 🔔"

            banner = (
                f"\n{c}{BOLD}╔══════════════════════════════════════════════════════════════════════════╗{RESET}\n"
                f"{c}{BOLD}║  {tag:<68}║{RESET}\n"
                f"{c}{BOLD}╠══════════════════════════════════════════════════════════════════════════╣{RESET}\n"
                f"{ev_json_str}\n"
                f"{c}{BOLD}╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n"
            )
            logger.state(banner)

    def _build_live_status(
        self,
        t: float,
        instant_state: str,
        instant_conf: float,
        obs: FrameObservation,
    ) -> Dict[str, Any]:
        """Constructs live state telemetry for real-time video overlay and external stream monitoring."""
        banner = None
        if self.last_event_banner and (t - self.last_event_banner_time <= 6.0):
            banner = self.last_event_banner

        return {
            "t": t,
            "live_state": self.active_state or instant_state,
            "live_conf": float(np.mean(self.active_confs)) if self.active_confs else instant_conf,
            "instant_state": instant_state,
            "instant_conf": instant_conf,
            "bed_overlap": obs.bed_overlap,
            "active_event_banner": banner,
            "total_events_so_far": len(self.final_events),
        }

    def finalize(self, total_duration_sec: float) -> Tuple[List[StateSegment], List[BedEvent]]:
        """
        Finalizes stream execution upon end-of-stream, ensuring 100% exact duration conservation.
        """
        if self.active_state is not None:
            # If pending state was active and lasted until stream end, check if it forms a final segment
            if self.pending_state is not None and (total_duration_sec - self.pending_start_t) >= self.min_segment_sec:
                # Commit previous active
                prev_conf = float(np.mean(self.active_confs)) if self.active_confs else 0.50
                self.finalized_segments.append(
                    StateSegment(
                        start_t=self.active_start_t,
                        end_t=self.pending_start_t,
                        state=self.active_state,
                        confidence=prev_conf,
                    )
                )
                # Commit final pending
                pend_conf = float(np.mean(self.pending_confs)) if self.pending_confs else 0.50
                self.finalized_segments.append(
                    StateSegment(
                        start_t=self.pending_start_t,
                        end_t=total_duration_sec,
                        state=self.pending_state,
                        confidence=pend_conf,
                    )
                )
            else:
                # Close active segment to end of stream
                final_conf = float(np.mean(self.active_confs)) if self.active_confs else 0.50
                self.finalized_segments.append(
                    StateSegment(
                        start_t=self.active_start_t,
                        end_t=total_duration_sec,
                        state=self.active_state,
                        confidence=final_conf,
                    )
                )

        # Ensure exact timeline continuity (§2 & §3)
        for k in range(len(self.finalized_segments) - 1):
            self.finalized_segments[k].end_t = self.finalized_segments[k + 1].start_t

        return self.finalized_segments, self.final_events


__all__ = ["StreamingProcessor"]
