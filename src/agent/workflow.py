"""
LangGraph StateGraph Workflow for BedSense AI Agentic Reasoning (§5 of Design Spec).
"""

import json
import logging
from typing import Any, Dict, List, Optional
from langgraph.graph import END, START, StateGraph
import numpy as np

from ..constants import (
    ALL_STATES,
    DECISION_ALERT,
    DECISION_MONITOR,
    DECISION_NORMAL,
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_UNKNOWN,
)
from ..contracts import FrameObservation, StateSegment
from .client import OllamaClient
from .state import AgentGraphState
from .tools import AgentToolSuite

logger = logging.getLogger("BedSenseAgent")


class LangGraphAgentWorkflow:
    """Constructs and executes the LangGraph state machine for ambiguous event reasoning."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama = ollama_client or OllamaClient()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)

        workflow.add_node("reasoning_node", self._reasoning_node)
        workflow.add_node("tools_node", self._tools_node)

        workflow.add_edge(START, "reasoning_node")

        workflow.add_conditional_edges(
            "reasoning_node",
            self._check_evidence_decision,
            {
                "tools": "tools_node",
                "end": END,
            },
        )
        workflow.add_edge("tools_node", "reasoning_node")

        return workflow.compile()

    # --- Graph Nodes ---
    def _reasoning_node(self, state: AgentGraphState) -> Dict[str, Any]:
        """Qwen Agent evaluates ambiguous segment + context history."""
        seg = state["segment"]
        ctx = state["context_history"]
        tool_count = state["tool_call_count"]
        max_tools = state.get("max_tool_calls")
        tool_trace = state["tool_trace"]

        cap_str = f"/{max_tools}" if max_tools is not None else ""
        logger.agent(
            f"[Graph: reasoning_node] Step {tool_count}{cap_str} | Evaluating segment [{seg.start_t:.1f}s - {seg.end_t:.1f}s] "
            f"State={seg.state} (Conf={seg.confidence*100:.0f}%) | Ambiguity Trigger: {seg.ambiguous_reason or 'None'}"
        )

        # Hard cap reached (only if max_tool_calls is explicitly set) -> emit best effort decision capped at 0.50
        if max_tools is not None and tool_count >= max_tools:
            logger.agent(f"[Graph: reasoning_node] Tool call cap reached ({max_tools}). Emitting bounded decision.")
            return {
                "needs_more_evidence": False,
                "final_state": seg.state if seg.state != STATE_UNKNOWN else STATE_UNKNOWN,
                "final_confidence": min(seg.confidence, 0.50),
                "final_event": None,
                "decision": DECISION_MONITOR if seg.state == STATE_UNKNOWN else DECISION_NORMAL,
                "reasoning": f"Tool call limit ({max_tools}) reached; emitting bounded decision.",
            }

        # Format context for LLM prompt (strictly filtered to 5.0s surrounding window)
        context_window_sec = 5.0
        filtered_ctx = [
            s for s in ctx
            if s.end_t >= (seg.start_t - context_window_sec) and s.start_t <= (seg.end_t + context_window_sec)
        ]
        ctx_summary = [f"[{s.start_t:.1f}s-{s.end_t:.1f}s] {s.state} (conf: {s.confidence:.2f})" for s in filtered_ctx]

        # Format complete tool execution trace history for the LLM
        trace_summary = json.dumps(
            [
                {
                    "step": t.get("step"),
                    "tool": t.get("tool"),
                    "args": t.get("args"),
                    "result": t.get("result"),
                }
                for t in tool_trace
            ],
            indent=2,
        ) if tool_trace else "None"

        system_prompt = (
            "You are an expert AI clinical monitoring agent for elderly bed safety. "
            "You analyze ambiguous temporal state segments and decide whether more evidence is needed (via tool calls) "
            "or if the situation can be resolved into a confirmed state and clinical decision (NORMAL, MONITOR, ALERT).\n\n"
            "Clinical Alert Decision Rules:\n"
            "• 'NORMAL': Person is lying, sitting, standing, or walking normally.\n"
            "• 'MONITOR': Initial bed exit ('bed_exit'), bed return ('bed_return'), person sits on bed edge, or state cannot be confidently determined (UNKNOWN).\n"
            "• 'ALERT': Safety event occurs, such as patient missing / not detected ('missing', out_of_bed -> unknown), prolonged absence from bed (> 10s), or horizontal lying off-bed (fall risk).\n\n"
            "Tool Selection Conditions:\n"
            "• 'check_bed_overlap': Use when you need the exact raw polygon overlap with the bed (0.0 = completely off bed, 1.0 = fully in bed) at a specific timestamp.\n"
            "• 'get_segment': Use when you need surrounding temporal state transitions within a time window [t0, t1].\n"
            "• 'vlm_describe': Use when keypoint pose confidence is low, resident position is obscured, or visual verification from video frames is required.\n\n"
            "Rules:\n"
            "1. Carefully inspect 'Executed Tool History'. Do NOT request a tool if the required evidence has already been retrieved.\n"
            "2. When you have sufficient evidence to resolve the ambiguity, set 'needs_more_evidence': false and supply final_state, confidence_score (reflecting your certainty in the clinical assessment), final_event, and decision."
        )
        user_prompt = (
            f"Ambiguous Segment: [{seg.start_t:.1f}s - {seg.end_t:.1f}s] State: {seg.state} (Raw Vision Conf: {seg.confidence:.2f})\n"
            f"Ambiguity Trigger: {seg.ambiguous_reason or 'None'}\n"
            f"Recent Context (5s window): {', '.join(ctx_summary)}\n"
            f"Executed Tool History:\n{trace_summary}\n\n"
            f"Decide: Do you need more evidence? If YES, specify one tool call: 'get_segment', 'check_bed_overlap', or 'vlm_describe'.\n"
            f"If NO, specify final_state, confidence_score (your confidence score for this reasoning, e.g. 0.85-0.95), final_event ('bed_exit', 'bed_return', 'missing', or null), and decision ('NORMAL', 'MONITOR', 'ALERT').\n"
            f"Respond ONLY in valid JSON matching:\n"
            f'{{"needs_more_evidence": bool, "tool_name": str|null, "tool_args": dict|null, "final_state": str|null, "confidence_score": float|null, "final_event": str|null, "decision": str|null, "reasoning": str}}'
        )

        llm_res = None
        if self.ollama.is_available():
            llm_res = self.ollama.generate_json(user_prompt, system_prompt)

        if llm_res and isinstance(llm_res, dict):
            needs_more = llm_res.get("needs_more_evidence", False)
            if needs_more and llm_res.get("tool_name"):
                t_name = llm_res.get("tool_name")
                t_args = llm_res.get("tool_args")
                if not isinstance(t_args, dict):
                    t_args = {}
                t_reason = llm_res.get("reasoning", "Agent requested tool evidence.")
                logger.agent(f"[Graph: reasoning_node] LLM requested tool: '{t_name}' args={t_args} | Rationale: {t_reason}")
                return {
                    "needs_more_evidence": True,
                    "next_tool_call": {
                        "tool": t_name,
                        "args": t_args,
                    },
                    "reasoning": t_reason,
                }
            else:
                f_state = llm_res.get("final_state") or seg.state
                raw_conf = llm_res.get("confidence_score") if llm_res.get("confidence_score") is not None else llm_res.get("final_confidence")
                if raw_conf is not None:
                    try:
                        f_conf = float(raw_conf)
                        f_conf = float(np.clip(f_conf, 0.05, 0.99))
                    except (ValueError, TypeError):
                        f_conf = float(seg.confidence)
                else:
                    f_conf = float(seg.confidence)
                f_event = llm_res.get("final_event")
                decision = llm_res.get("decision") or DECISION_MONITOR
                f_reason = llm_res.get("reasoning", "Resolved directly from context.")
                logger.agent(f"[Graph: reasoning_node] LLM resolved decision -> State: {f_state}, Event: {f_event}, Decision: {decision}, Agent Confidence: {f_conf*100:.0f}% | Reason: {f_reason}")
                return {
                    "needs_more_evidence": False,
                    "final_state": f_state if f_state in ALL_STATES else seg.state,
                    "final_confidence": f_conf,
                    "confidence_score": f_conf,
                    "final_event": f_event,
                    "decision": decision,
                    "reasoning": f_reason,
                }

        # LLM offline / unavailable: direct passthrough without heuristics
        logger.agent(f"[Graph: reasoning_node] LLM offline/unavailable — retaining original segment state '{seg.state}' (Conf: {seg.confidence*100:.0f}%) without heuristics.")
        return {
            "needs_more_evidence": False,
            "final_state": seg.state,
            "final_confidence": seg.confidence,
            "final_event": None,
            "decision": DECISION_NORMAL if seg.state == STATE_LYING_IN_BED else DECISION_MONITOR,
            "reasoning": "Direct passthrough from temporal segment (LLM offline).",
        }

    def _tools_node(self, state: AgentGraphState) -> Dict[str, Any]:
        """Executes requested tool and records trace for auditability (§2 BedEvent contract)."""
        tool_call = state.get("next_tool_call") or {}
        tool_name = tool_call.get("tool", "")
        args = tool_call.get("args")
        if not isinstance(args, dict):
            args = {}

        result: Any = None
        step_idx = state["tool_call_count"] + 1
        logger.agent(f"[Graph: tools_node] Executing step {step_idx}: tool='{tool_name}' args={args}")

        if tool_name == "get_segment":
            t0 = float(args.get("t0") if args.get("t0") is not None else max(0.0, state["segment"].start_t - 5.0))
            t1 = float(args.get("t1") if args.get("t1") is not None else (state["segment"].end_t + 5.0))
            result = AgentToolSuite.get_segment(t0, t1, state["context_history"])
        elif tool_name == "check_bed_overlap":
            t = float(args.get("t") if args.get("t") is not None else ((state["segment"].start_t + state["segment"].end_t) / 2.0))
            result = AgentToolSuite.check_bed_overlap(t, state["observations"])
        elif tool_name == "vlm_describe":
            t0 = float(args.get("t0") if args.get("t0") is not None else state["segment"].start_t)
            t1 = float(args.get("t1") if args.get("t1") is not None else state["segment"].end_t)
            q = str(args.get("question") or "What is the resident doing?")
            result = AgentToolSuite.vlm_describe(t0, t1, q, state.get("video_path"), self.ollama)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        logger.agent(f"[Graph: tools_node] Completed step {step_idx}: '{tool_name}' -> Result: {result}")

        new_trace = list(state["tool_trace"])
        new_trace.append({
            "step": step_idx,
            "tool": tool_name,
            "args": args,
            "result": result,
        })

        return {
            "tool_trace": new_trace,
            "tool_call_count": step_idx,
            "last_tool_result": result,
            "next_tool_call": None,
        }

    def _check_evidence_decision(self, state: AgentGraphState) -> str:
        """Conditional routing: continue with tools or conclude."""
        max_tools = state.get("max_tool_calls")
        has_cap = max_tools is not None and max_tools > 0
        if state.get("needs_more_evidence", False) and (not has_cap or state["tool_call_count"] < max_tools):
            logger.agent(f"[Graph: Router] Transition -> 'tools_node' (calls made: {state['tool_call_count']})")
            return "tools"
        logger.agent("[Graph: Router] Transition -> 'END' (Reasoning complete)")
        return "end"

    def run(
        self,
        segment: StateSegment,
        context_history: List[StateSegment],
        observations: List[FrameObservation],
        video_path: Optional[str] = None,
        max_tool_calls: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the LangGraph agent on an ambiguous segment."""
        logger.agent(
            f"--- Starting LangGraph Agent Workflow for Segment [{segment.start_t:.1f}s - {segment.end_t:.1f}s] "
            f"(State: {segment.state}, Conf: {segment.confidence*100:.0f}%) ---"
        )
        initial_state: AgentGraphState = {
            "segment": segment,
            "context_history": context_history,
            "observations": observations,
            "video_path": video_path,
            "tool_trace": [],
            "tool_call_count": 0,
            "max_tool_calls": max_tool_calls,
            "needs_more_evidence": False,
            "next_tool_call": None,
            "last_tool_result": None,
            "final_state": segment.state,
            "final_confidence": segment.confidence,
            "final_event": None,
            "decision": DECISION_NORMAL,
            "reasoning": "",
        }

        output_state = self.graph.invoke(initial_state)
        logger.agent(
            f"--- Finished LangGraph Agent Workflow: Decision={output_state.get('decision')}, "
            f"FinalState={output_state.get('final_state')}, Conf={output_state.get('final_confidence', 0)*100:.0f}%, "
            f"Tools Used={len(output_state.get('tool_trace', []))} ---"
        )
        return {
            "final_state": output_state.get("final_state", segment.state),
            "final_confidence": output_state.get("final_confidence", segment.confidence),
            "final_event": output_state.get("final_event"),
            "decision": output_state.get("decision", DECISION_NORMAL),
            "reasoning": output_state.get("reasoning", ""),
            "tool_trace": output_state.get("tool_trace", []),
        }


__all__ = ["LangGraphAgentWorkflow"]
