"""
Agent Graph State schema for BedSense AI LangGraph Reasoning Layer.
"""

from typing import Any, Dict, List, Optional, TypedDict
from ..contracts import FrameObservation, StateSegment


class AgentGraphState(TypedDict):
    """LangGraph execution state passed through reasoning and tool execution nodes."""
    segment: StateSegment
    context_history: List[StateSegment]
    observations: List[FrameObservation]
    video_path: Optional[str]
    tool_trace: List[Dict[str, Any]]
    tool_call_count: int
    max_tool_calls: Optional[int]
    needs_more_evidence: bool
    next_tool_call: Optional[Dict[str, Any]]
    last_tool_result: Optional[Any]
    final_state: str
    final_confidence: float
    final_event: Optional[str]
    decision: str
    reasoning: str


__all__ = ["AgentGraphState"]
