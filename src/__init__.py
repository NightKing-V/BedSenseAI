"""
BedSense AI — Vision -> Temporal Processing -> State Mechanism Monitoring System.
"""

from .agent import (
    AgentGraphState,
    AgentToolSuite,
    LangGraphAgentWorkflow,
    OllamaClient,
)
from .constants import (
    ALL_STATES,
    DECISION_ALERT,
    DECISION_MONITOR,
    DECISION_NORMAL,
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    EVENT_MISSING,
    STATE_COLORS,
    STATE_LYING_IN_BED,
    STATE_OUT_OF_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_UNKNOWN,
    STATE_WALKING,
)
from .logging_config import (
    AGENT_LEVEL_NUM,
    STATE_LEVEL_NUM,
    BedSenseFormatter,
    load_config,
    setup_logger,
)
from .evidence import (
    ActivityDecision,
    ActivityEvidence,
    EvidenceTemporalClassifier,
    RuleBasedEvidenceExtractor,
)
from .policy import (
    PolicyEngine,
    ReportGenerator,
    evaluate_ambiguity,
    format_hms,
)
from .vision import BedRelationEngine, PerceptionDetector

__all__ = [
    # Vision Layer
    "BedRelationEngine",
    "PerceptionDetector",
    # Streaming Layer
    "StreamingProcessor",
    # Evidence Classification Layer
    "EvidenceTemporalClassifier",
    "RuleBasedEvidenceExtractor",
    "ActivityEvidence",
    "ActivityDecision",
    # LangGraph Agent Layer
    "LangGraphAgentWorkflow",
    "OllamaClient",
    "AgentToolSuite",
    "AgentGraphState",
    # State Mechanism & Policy Layer
    "BedPatternMatcher",
    "CandidateEvent",
    "PolicyEngine",
    "ReportGenerator",
    "format_hms",
    # Contracts
    "FrameObservation",
    "StateSegment",
    "BedEvent",
    # Logging & Config
    "setup_logger",
    "load_config",
    "BedSenseFormatter",
    "AGENT_LEVEL_NUM",
    "STATE_LEVEL_NUM",
    # Constants
    "ALL_STATES",
    "STATE_LYING_IN_BED",
    "STATE_SITTING_ON_BED",
    "STATE_SITTING_OUTSIDE_BED",
    "STATE_STANDING",
    "STATE_WALKING",
    "STATE_OUT_OF_BED",
    "STATE_UNKNOWN",
    "EVENT_BED_EXIT",
    "EVENT_BED_RETURN",
    "EVENT_MISSING",
    "DECISION_NORMAL",
    "DECISION_MONITOR",
    "DECISION_ALERT",
    "STATE_COLORS",
]
